import csv
import zarr
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import OrderedDict
import logging

from .global_index_manager import GlobalIndexManager
from ..perf import Timer
from ..tensor_transforms import InstanceNorm, NoTransform

logger = logging.getLogger(__name__)


@dataclass
class CheckpointableStreamDatasetConfig:
    """
    Configuration for CheckpointableStreamDataset.
    
    Attributes:
        path_csv (str): CSV file to configure the dataset list
        transforms (List): A list of transformations to apply to each data item
        buffer_size (int): The number of zarr files buffered in memory (LRU cache)
        dist_rank (int): Current distributed process rank
        dist_world_size (int): Total number of distributed processes
        device (str): Device string (e.g., 'cuda:0')
        dtype (torch.dtype, optional): The torch.dtype for representing data
        uses_norm (bool): Whether to apply instance normalization
        scales_variance (bool): Whether to scale variance in normalization
        perfs_runtime (bool): Whether to measure performance timing
        global_index_cache (str, optional): Path to global index cache file
    """
    path_csv: str
    transforms: List
    buffer_size: int
    dist_rank: int
    dist_world_size: int
    device: str
    dtype: Optional[torch.dtype] = None
    uses_norm: bool = True
    scales_variance: bool = True
    perfs_runtime: bool = False
    global_index_cache: Optional[str] = None


class CheckpointableStreamDataset(Dataset):
    """
    Dataset with global sample indexing for precise checkpoint resumption.
    
    Replaces SegmentedPeakNetDataset with a simpler, more flexible approach:
    - Global sample indexing across all zarr files
    - Sample-level checkpoint resumption
    - Variable GPU count support
    - Standard PyTorch Dataset interface for DataLoader compatibility
    """
    
    def __init__(self, config: CheckpointableStreamDatasetConfig):
        super().__init__()
        
        self.config = config
        self.dist_rank = config.dist_rank
        self.dist_world_size = config.dist_world_size
        self.transforms = config.transforms
        self.buffer_size = config.buffer_size
        self.dtype = config.dtype
        self.uses_norm = config.uses_norm
        self.scales_variance = config.scales_variance
        self.perfs_runtime = config.perfs_runtime
        
        # Initialize normalization
        self.norm = InstanceNorm(scales_variance=self.scales_variance) if self.uses_norm else NoTransform()
        
        # Load file paths from CSV
        self.file_paths = self._load_file_paths(config.path_csv)
        logger.info(f"Loaded {len(self.file_paths)} file paths from {config.path_csv}")
        
        # Initialize global index manager
        self.global_manager = GlobalIndexManager(
            self.file_paths, 
            cache_path=config.global_index_cache
        )
        
        # Calculate this rank's sample range
        self.rank_start_idx, self.rank_end_idx = self.global_manager.get_rank_sample_range(
            self.dist_world_size, self.dist_rank
        )
        
        # Checkpoint state - tracks progress within this rank's range
        self.local_samples_processed = 0  # How many samples this rank has processed
        
        # Zarr file caching (LRU cache like original implementation)
        self.zarr_cache = OrderedDict()
        
        logger.info(f"Rank {self.dist_rank}: Assigned global samples "
                   f"[{self.rank_start_idx}, {self.rank_end_idx}) "
                   f"({self.rank_end_idx - self.rank_start_idx} samples)")
    
    def _load_file_paths(self, path_csv: str) -> List[str]:
        """Load file paths from CSV file"""
        file_paths = []
        try:
            with open(path_csv, 'r') as fh:
                lines = list(csv.reader(fh))
            file_paths = [line[0] for line in lines if line]  # Skip empty lines
        except Exception as e:
            logger.error(f"Failed to load file paths from {path_csv}: {e}")
            raise
        return file_paths
    
    def _get_cached_zarr(self, file_path: str):
        """Get zarr data with LRU caching"""
        if file_path not in self.zarr_cache:
            # Evict oldest if cache is full
            if len(self.zarr_cache) >= self.buffer_size:
                oldest_path, _ = self.zarr_cache.popitem(last=False)
                logger.debug(f"Evicted {oldest_path} from zarr cache")
            
            # Load new file
            try:
                self.zarr_cache[file_path] = zarr.open(file_path, mode='r')
                logger.debug(f"Loaded {file_path} into zarr cache")
            except Exception as e:
                logger.error(f"Failed to load zarr file {file_path}: {e}")
                raise
        else:
            # Move to end (most recently used)
            self.zarr_cache.move_to_end(file_path)
        
        return self.zarr_cache[file_path]
    
    def __len__(self) -> int:
        """Return number of samples assigned to this rank"""
        return self.rank_end_idx - self.rank_start_idx - self.local_samples_processed
    
    def __getitem__(self, idx: int):
        """
        Get sample by local index within this rank's assignment.
        
        Args:
            idx: Local index within this rank (0 to len(self)-1)
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Convert local index to global index
        global_idx = self.rank_start_idx + self.local_samples_processed + idx
        
        # Get file and item index
        file_path, item_idx = self.global_manager.global_to_file_index(global_idx)
        
        # Load data from zarr file
        data = self._get_cached_zarr(file_path)
        
        # Extract image and label tensors
        image = torch.from_numpy(data.get("images")[item_idx][None, None, :, :])
        label = torch.from_numpy(data.get("labels")[item_idx][None, None, :, :])
        
        # Apply transformations
        if self.transforms is not None:
            data_combined = torch.cat([image, label], dim=0)  # (2*B=1, C, H, W)
            
            if self.dtype is not None:
                data_combined = data_combined.to(self.dtype)
            
            for enum_idx, transform in enumerate(self.transforms):
                if self.perfs_runtime:
                    with Timer(tag=f"Transform method {enum_idx:d}", is_on=True):
                        data_combined = transform(data_combined)
                else:
                    data_combined = transform(data_combined)
            
            image = data_combined[0]  # (1, C, H, W)
            label = data_combined[1]  # (1, C, H, W)
            
            # Binarize the label
            label = label > 0.5
        
        # Apply normalization to image
        image = self.norm(image[None, :])[0]  # (C, H, W)
        
        return image, label  # (C, H, W)
    
    def advance_local_progress(self, num_samples: int):
        """
        Advance local progress counter.
        
        This should be called by the training loop to track progress
        for checkpoint purposes.
        
        Args:
            num_samples: Number of samples processed
        """
        self.local_samples_processed += num_samples
        
        if self.local_samples_processed > (self.rank_end_idx - self.rank_start_idx):
            logger.warning(f"Local progress {self.local_samples_processed} exceeds "
                          f"assigned range {self.rank_end_idx - self.rank_start_idx}")
    
    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get checkpoint state for this rank.
        
        Returns:
            Dict containing rank's checkpoint state
        """
        assigned_samples = self.rank_end_idx - self.rank_start_idx
        progress_percent = (self.local_samples_processed / assigned_samples * 100) if assigned_samples > 0 else 100.0
        
        return {
            'dist_rank': self.dist_rank,
            'dist_world_size': self.dist_world_size,
            'rank_start_idx': self.rank_start_idx,
            'rank_end_idx': self.rank_end_idx,
            'local_samples_processed': self.local_samples_processed,
            'global_samples_processed': self.rank_start_idx + self.local_samples_processed,
            'total_global_samples': self.global_manager.total_samples,
            'progress_percent': progress_percent
        }
    
    def restore_checkpoint_state(self, checkpoint_state: Dict[str, Any]):
        """
        Restore from checkpoint state.
        
        Args:
            checkpoint_state: Dict containing checkpoint state
        """
        self.local_samples_processed = checkpoint_state.get('local_samples_processed', 0)
        
        logger.info(f"Restored checkpoint state: rank {self.dist_rank}, "
                   f"local progress {self.local_samples_processed}")
    
    def create_for_new_world_size(self, new_world_size: int, new_rank: int, 
                                  global_progress: int) -> 'CheckpointableStreamDataset':
        """
        Create new dataset instance for different world size during resumption.
        
        Args:
            new_world_size: New number of ranks
            new_rank: New rank for this process
            global_progress: Total global samples processed so far
            
        Returns:
            New CheckpointableStreamDataset instance configured for new world size
        """
        # Create new config with updated world size and rank
        new_config = CheckpointableStreamDatasetConfig(
            path_csv=self.config.path_csv,
            transforms=self.config.transforms,
            buffer_size=self.config.buffer_size,
            dist_rank=new_rank,
            dist_world_size=new_world_size,
            device=self.config.device,
            dtype=self.config.dtype,
            uses_norm=self.config.uses_norm,
            scales_variance=self.config.scales_variance,
            perfs_runtime=self.config.perfs_runtime,
            global_index_cache=self.config.global_index_cache
        )
        
        # Create new dataset instance
        new_dataset = CheckpointableStreamDataset(new_config)
        
        # Calculate new work distribution
        remaining_ranges = new_dataset.global_manager.calculate_remaining_work(
            global_progress, new_world_size
        )
        
        # Set new rank's range
        if new_rank < len(remaining_ranges):
            new_dataset.rank_start_idx, new_dataset.rank_end_idx = remaining_ranges[new_rank]
            new_dataset.local_samples_processed = 0  # Start fresh in new range
        else:
            # No work for this rank
            new_dataset.rank_start_idx = new_dataset.global_manager.total_samples
            new_dataset.rank_end_idx = new_dataset.global_manager.total_samples
            new_dataset.local_samples_processed = 0
        
        logger.info(f"Created dataset for new world size {new_world_size}, "
                   f"rank {new_rank}: global range [{new_dataset.rank_start_idx}, "
                   f"{new_dataset.rank_end_idx})")
        
        return new_dataset
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get detailed progress information for monitoring"""
        return {
            'rank': self.dist_rank,
            'world_size': self.dist_world_size,
            'assigned_samples': self.rank_end_idx - self.rank_start_idx,
            'processed_samples': self.local_samples_processed,
            'remaining_samples': len(self),
            'progress_percent': (self.local_samples_processed / 
                               (self.rank_end_idx - self.rank_start_idx) * 100) 
                              if self.rank_end_idx > self.rank_start_idx else 100.0,
            'global_start': self.rank_start_idx,
            'global_end': self.rank_end_idx,
            'global_current': self.rank_start_idx + self.local_samples_processed
        }
    
    def reset_progress(self):
        """Reset progress counter (for new epoch)"""
        self.local_samples_processed = 0
        logger.info(f"Reset progress for rank {self.dist_rank}")
    
    def validate_dataset(self) -> bool:
        """Validate dataset configuration and global index"""
        try:
            # Validate global index
            if not self.global_manager.validate_index():
                return False
            
            # Check rank assignment
            if self.rank_start_idx >= self.rank_end_idx:
                logger.warning(f"Rank {self.dist_rank} has no samples assigned")
            
            # Test accessing a few samples
            if len(self) > 0:
                test_indices = [0]
                if len(self) > 1:
                    test_indices.append(len(self) - 1)
                if len(self) > 2:
                    test_indices.append(len(self) // 2)
                
                for idx in test_indices:
                    try:
                        image, label = self[idx]
                        if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
                            logger.error(f"Invalid sample type at index {idx}")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to access sample {idx}: {e}")
                        return False
            
            logger.info(f"Dataset validation passed for rank {self.dist_rank}")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False