import os
import pickle
import zarr
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class GlobalIndexManager:
    """
    Manages global sample indexing across multiple zarr files for precise checkpointing.
    
    Creates a virtual global dataset where every sample gets a unique global index,
    enabling sample-level checkpoint resumption and variable GPU count support.
    """
    
    def __init__(self, file_paths: List[str], cache_path: Optional[str] = None):
        """
        Initialize global index manager.
        
        Args:
            file_paths: List of zarr file paths in the dataset
            cache_path: Path to cache file. If None, uses auto-generated name
        """
        self.file_paths = file_paths
        self.cache_path = cache_path or self._generate_cache_path()
        
        # Index mappings
        self.file_lengths: Dict[str, int] = {}
        self.cumulative_lengths: List[int] = []
        self.total_samples: int = 0
        
        # Build or load index
        if self._is_cache_valid():
            logger.info(f"Loading global index from cache: {self.cache_path}")
            self._load_from_cache()
        else:
            logger.info("Building global index from zarr files")
            self._build_global_index()
            self._save_to_cache()
            
        logger.info(f"Global dataset: {self.total_samples} samples across {len(self.file_paths)} files")
    
    def _generate_cache_path(self) -> str:
        """Generate cache path based on file paths hash"""
        import hashlib
        file_paths_str = '|'.join(sorted(self.file_paths))
        hash_obj = hashlib.md5(file_paths_str.encode())
        return f"global_index_cache_{hash_obj.hexdigest()[:8]}.pkl"
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is newer than all zarr files"""
        if not os.path.exists(self.cache_path):
            return False
            
        try:
            cache_mtime = os.path.getmtime(self.cache_path)
            for file_path in self.file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    return False
                if os.path.getmtime(file_path) > cache_mtime:
                    logger.info(f"File {file_path} newer than cache, rebuilding index")
                    return False
            return True
        except OSError as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def _build_global_index(self):
        """Build global index by scanning all zarr files"""
        self.file_lengths = {}
        self.cumulative_lengths = []
        total = 0
        
        for i, file_path in enumerate(self.file_paths):
            try:
                data = zarr.open(file_path, mode='r')
                file_length = data['images'].shape[0]
                self.file_lengths[file_path] = file_length
                self.cumulative_lengths.append(total)
                total += file_length
                
                if i % 100 == 0 or i == len(self.file_paths) - 1:
                    logger.info(f"Processed {i+1}/{len(self.file_paths)} files, "
                              f"total samples: {total}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Set length to 0 for problematic files
                self.file_lengths[file_path] = 0
                self.cumulative_lengths.append(total)
        
        self.total_samples = total
    
    def _save_to_cache(self):
        """Save global index to cache file"""
        try:
            cache_data = {
                'file_paths': self.file_paths,
                'file_lengths': self.file_lengths,
                'cumulative_lengths': self.cumulative_lengths,
                'total_samples': self.total_samples
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Global index cached to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_cache(self):
        """Load global index from cache file"""
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data matches current file paths
            if cache_data['file_paths'] != self.file_paths:
                logger.warning("Cache file paths don't match, rebuilding index")
                self._build_global_index()
                return
            
            self.file_lengths = cache_data['file_lengths']
            self.cumulative_lengths = cache_data['cumulative_lengths']
            self.total_samples = cache_data['total_samples']
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, rebuilding index")
            self._build_global_index()
    
    def global_to_file_index(self, global_idx: int) -> Tuple[str, int]:
        """
        Convert global sample index to (file_path, item_idx).
        
        Args:
            global_idx: Global sample index
            
        Returns:
            Tuple of (file_path, item_index_within_file)
            
        Raises:
            IndexError: If global_idx is out of range
        """
        if global_idx < 0 or global_idx >= self.total_samples:
            raise IndexError(f"Global index {global_idx} out of range [0, {self.total_samples})")
        
        # Binary search for efficiency with large number of files
        left, right = 0, len(self.cumulative_lengths) - 1
        
        while left <= right:
            mid = (left + right) // 2
            cumulative_start = self.cumulative_lengths[mid]
            file_length = self.file_lengths[self.file_paths[mid]]
            cumulative_end = cumulative_start + file_length
            
            if cumulative_start <= global_idx < cumulative_end:
                local_idx = global_idx - cumulative_start
                return self.file_paths[mid], local_idx
            elif global_idx < cumulative_start:
                right = mid - 1
            else:
                left = mid + 1
        
        # Fallback to linear search (shouldn't happen with correct implementation)
        for i, cumulative_start in enumerate(self.cumulative_lengths):
            file_length = self.file_lengths[self.file_paths[i]]
            if cumulative_start <= global_idx < cumulative_start + file_length:
                local_idx = global_idx - cumulative_start
                return self.file_paths[i], local_idx
        
        raise IndexError(f"Failed to map global index {global_idx}")
    
    def get_rank_sample_range(self, world_size: int, rank: int) -> Tuple[int, int]:
        """
        Calculate global sample range for a specific rank.
        
        Args:
            world_size: Total number of ranks
            rank: Current rank (0-indexed)
            
        Returns:
            Tuple of (start_global_idx, end_global_idx) exclusive end
        """
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Rank {rank} out of range [0, {world_size})")
        
        samples_per_rank = self.total_samples // world_size
        remainder = self.total_samples % world_size
        
        # Distribute remainder across first few ranks
        if rank < remainder:
            start_idx = rank * (samples_per_rank + 1)
            end_idx = start_idx + samples_per_rank + 1
        else:
            start_idx = remainder * (samples_per_rank + 1) + (rank - remainder) * samples_per_rank
            end_idx = start_idx + samples_per_rank
        
        return start_idx, end_idx
    
    def calculate_remaining_work(self, global_progress: int, new_world_size: int) -> List[Tuple[int, int]]:
        """
        Calculate work distribution for resumption with different world size.
        
        Args:
            global_progress: Total samples processed so far
            new_world_size: New number of ranks
            
        Returns:
            List of (start_idx, end_idx) ranges for each new rank
        """
        remaining_samples = self.total_samples - global_progress
        
        if remaining_samples <= 0:
            return [(global_progress, global_progress) for _ in range(new_world_size)]
        
        samples_per_rank = remaining_samples // new_world_size
        remainder = remaining_samples % new_world_size
        
        ranges = []
        current_start = global_progress
        
        for rank in range(new_world_size):
            if rank < remainder:
                rank_samples = samples_per_rank + 1
            else:
                rank_samples = samples_per_rank
                
            rank_end = current_start + rank_samples
            ranges.append((current_start, rank_end))
            current_start = rank_end
        
        return ranges
    
    def get_file_stats(self) -> Dict[str, int]:
        """Get file length statistics for debugging"""
        return dict(self.file_lengths)
    
    def validate_index(self) -> bool:
        """Validate global index consistency"""
        try:
            # Check total samples calculation
            calculated_total = sum(self.file_lengths.values())
            if calculated_total != self.total_samples:
                logger.error(f"Total samples mismatch: {calculated_total} vs {self.total_samples}")
                return False
            
            # Check cumulative lengths
            if len(self.cumulative_lengths) != len(self.file_paths):
                logger.error("Cumulative lengths count mismatch")
                return False
            
            # Test a few random global indices
            import random
            test_indices = random.sample(range(self.total_samples), min(100, self.total_samples))
            
            for global_idx in test_indices:
                try:
                    file_path, item_idx = self.global_to_file_index(global_idx)
                    if item_idx >= self.file_lengths[file_path]:
                        logger.error(f"Invalid item index {item_idx} for file {file_path}")
                        return False
                except Exception as e:
                    logger.error(f"Failed to map global index {global_idx}: {e}")
                    return False
            
            logger.info("Global index validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False