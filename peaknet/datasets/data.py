import csv
import zarr
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import OrderedDict
import logging
from einops import rearrange

from .indexing import GlobalIndexManager
from ..perf import Timer
from ..tensor_transforms import InstanceNorm, NoTransform

logger = logging.getLogger(__name__)


@dataclass
class PeakNetDatasetConfig:
    """
    Configuration for PeakNetDataset.

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
        enable_shuffling (bool): Whether to enable cross-rank shuffling
        shuffle_seed_base (int): Base seed for deterministic shuffling
        reshuffle_frequency (int): Steps between reshuffles (0 = no reshuffling)
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
    enable_shuffling: bool = False
    shuffle_seed_base: int = 42
    reshuffle_frequency: int = 0


class PeakNetDataset(Dataset):
    """
    Dataset with global sample indexing for precise checkpoint resumption.

    Replaces SegmentedPeakNetDataset with a simpler, more flexible approach:
    - Global sample indexing across all zarr files
    - Sample-level checkpoint resumption
    - Variable GPU count support
    - Standard PyTorch Dataset interface for DataLoader compatibility
    """

    def __init__(self, config: PeakNetDatasetConfig):
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

        # Shuffle configuration
        self.enable_shuffling = config.enable_shuffling
        self.shuffle_seed_base = config.shuffle_seed_base
        self.reshuffle_frequency = config.reshuffle_frequency
        self.current_shuffle_epoch = 0

        # Generate assigned global indices for this rank
        self.assigned_global_indices = self._generate_assigned_indices()

        # Checkpoint state - tracks progress within assigned indices
        self.local_samples_processed = 0  # How many samples this rank has processed

        # Zarr file caching (LRU cache like original implementation)
        self.zarr_cache = OrderedDict()

        logger.info(f"Rank {self.dist_rank}: Assigned {len(self.assigned_global_indices)} global indices "
                   f"(shuffling {'enabled' if self.enable_shuffling else 'disabled'})")

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

    def _generate_assigned_indices(self) -> List[int]:
        """Generate assigned global indices for this rank using shuffle-then-partition."""
        total_samples = self.global_manager.total_samples
        
        # Generate global index list
        global_indices = list(range(total_samples))
        
        if self.enable_shuffling:
            # Shuffle global indices using deterministic seed
            seed = self.shuffle_seed_base + self.current_shuffle_epoch
            import random
            rng = random.Random(seed)
            rng.shuffle(global_indices)
            logger.info(f"Rank {self.dist_rank}: Shuffled global indices with seed {seed} "
                       f"for shuffle epoch {self.current_shuffle_epoch}")
        
        # Partition shuffled indices among ranks
        samples_per_rank = total_samples // self.dist_world_size
        remainder = total_samples % self.dist_world_size
        
        # Calculate start and end for this rank's partition
        start_idx = self.dist_rank * samples_per_rank + min(self.dist_rank, remainder)
        end_idx = start_idx + samples_per_rank + (1 if self.dist_rank < remainder else 0)
        
        # Extract assigned indices for this rank
        assigned_indices = global_indices[start_idx:end_idx]
        
        logger.info(f"Rank {self.dist_rank}: Assigned {len(assigned_indices)} global indices "
                   f"from shuffled list (shuffle epoch {self.current_shuffle_epoch})")
        
        return assigned_indices

    def maybe_reshuffle(self, current_step: int):
        """
        Reshuffle assigned indices if it's time based on reshuffle_frequency.
        
        Args:
            current_step: Current training step
        """
        if (self.enable_shuffling and 
            self.reshuffle_frequency > 0 and 
            current_step > 0 and
            current_step % self.reshuffle_frequency == 0):
            self.current_shuffle_epoch += 1
            self.assigned_global_indices = self._generate_assigned_indices()
            logger.info(f"Rank {self.dist_rank}: Reshuffled at step {current_step}, "
                       f"new shuffle epoch {self.current_shuffle_epoch}")

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
        """Return number of remaining samples assigned to this rank"""
        return len(self.assigned_global_indices) - self.local_samples_processed

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

        # Get assigned global index for this local index
        assigned_idx = self.local_samples_processed + idx
        global_idx = self.assigned_global_indices[assigned_idx]

        # Get file and item index
        file_path, item_idx = self.global_manager.global_to_file_index(global_idx)

        # Load data from zarr file
        data = self._get_cached_zarr(file_path)

        # Extract image and label tensors from B x H x W zarr structure
        # Images and labels are stored as (B, H, W) in zarr, extract single sample
        image_np = data.get("images")[item_idx]  # Shape: (H, W)
        label_np = data.get("labels")[item_idx]  # Shape: (H, W)

        # Convert to tensors and add channel dimension: (H, W) -> (1, H, W)
        image = rearrange(torch.from_numpy(image_np), 'h w -> 1 h w')
        label = rearrange(torch.from_numpy(label_np), 'h w -> 1 h w')

        # Apply transformations
        if self.transforms is not None:
            # Stack image and label along channel dimension: (1,H,W) + (1,H,W) -> (2,H,W)
            data_combined = rearrange([image, label], 'n c h w -> (n c) h w')

            if self.dtype is not None:
                data_combined = data_combined.to(self.dtype)

            # Add batch dimension for transforms that expect 4D tensors (B, C, H, W)
            data_combined = rearrange(data_combined, 'c h w -> 1 c h w')  # (2, H, W) → (1, 2, H, W)

            for enum_idx, transform in enumerate(self.transforms):
                if self.perfs_runtime:
                    with Timer(tag=f"Transform method {enum_idx:d}", is_on=True):
                        data_combined = transform(data_combined)
                else:
                    data_combined = transform(data_combined)

            # Remove batch dimension after transforms
            data_combined = rearrange(data_combined, '1 c h w -> c h w')  # (1, 2, H, W) → (2, H, W)

            # Extract back to separate tensors and maintain channel dimension
            # Split the combined (2, H, W) back into two (1, H, W) tensors
            image, label = rearrange(data_combined, '(n c) h w -> n c h w', n=2)

            # Binarize the label
            label = label > 0.5

        # Apply normalization to image (requires batch dimension)
        image = rearrange(image, 'c h w -> 1 c h w')  # Add batch dimension
        image = self.norm(image)
        image = rearrange(image, '1 c h w -> c h w')  # Remove batch dimension

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

        if self.local_samples_processed > len(self.assigned_global_indices):
            logger.warning(f"Local progress {self.local_samples_processed} exceeds "
                          f"assigned indices {len(self.assigned_global_indices)}")

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get checkpoint state for this rank.

        Returns:
            Dict containing rank's checkpoint state
        """
        assigned_samples = len(self.assigned_global_indices)
        progress_percent = (self.local_samples_processed / assigned_samples * 100) if assigned_samples > 0 else 100.0

        return {
            'dist_rank': self.dist_rank,
            'dist_world_size': self.dist_world_size,
            'assigned_indices': self.assigned_global_indices.copy(),
            'local_samples_processed': self.local_samples_processed,
            'total_global_samples': self.global_manager.total_samples,
            'progress_percent': progress_percent,
            # Shuffle state
            'enable_shuffling': self.enable_shuffling,
            'shuffle_seed_base': self.shuffle_seed_base,
            'current_shuffle_epoch': self.current_shuffle_epoch
        }

    def restore_checkpoint_state(self, checkpoint_state: Dict[str, Any]):
        """
        Restore from checkpoint state.

        Args:
            checkpoint_state: Dict containing checkpoint state
        """
        self.local_samples_processed = checkpoint_state.get('local_samples_processed', 0)
        
        # Restore shuffle state and assigned indices if present
        if 'enable_shuffling' in checkpoint_state:
            self.enable_shuffling = checkpoint_state['enable_shuffling']
            self.shuffle_seed_base = checkpoint_state.get('shuffle_seed_base', self.shuffle_seed_base)
            self.current_shuffle_epoch = checkpoint_state.get('current_shuffle_epoch', 0)
            
            if 'assigned_indices' in checkpoint_state:
                self.assigned_global_indices = checkpoint_state['assigned_indices']
                logger.info(f"Restored checkpoint with {len(self.assigned_global_indices)} assigned indices: "
                           f"rank {self.dist_rank}, shuffle epoch {self.current_shuffle_epoch}")
            else:
                # Regenerate assigned indices if not saved
                self.assigned_global_indices = self._generate_assigned_indices()
        
        logger.info(f"Restored checkpoint state: rank {self.dist_rank}, "
                   f"local progress {self.local_samples_processed}")

    def create_for_new_world_size(self, new_world_size: int, new_rank: int, 
                                  global_progress: int) -> 'PeakNetDataset':
        """
        Create new dataset instance for different world size during resumption.

        Args:
            new_world_size: New number of ranks
            new_rank: New rank for this process
            global_progress: Total global samples processed so far

        Returns:
            New PeakNetDataset instance configured for new world size
        """
        # Create new config with updated world size and rank
        new_config = PeakNetDatasetConfig(
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
            global_index_cache=self.config.global_index_cache,
            enable_shuffling=self.config.enable_shuffling,
            shuffle_seed_base=self.config.shuffle_seed_base,
            reshuffle_frequency=self.config.reshuffle_frequency
        )

        # Create new dataset instance with same shuffle state
        new_dataset = PeakNetDataset(new_config)
        new_dataset.current_shuffle_epoch = self.current_shuffle_epoch
        
        # Generate remaining work assignment for new world size
        # Use the same shuffle state to ensure deterministic assignment
        total_samples = new_dataset.global_manager.total_samples
        remaining_samples = total_samples - global_progress
        
        if remaining_samples <= 0:
            # No work remaining
            new_dataset.assigned_global_indices = []
            new_dataset.local_samples_processed = 0
        else:
            # Generate global indices for remaining work
            global_indices = list(range(total_samples))
            
            if new_dataset.enable_shuffling:
                # Use same shuffle state for consistency
                seed = new_dataset.shuffle_seed_base + new_dataset.current_shuffle_epoch
                import random
                rng = random.Random(seed)
                rng.shuffle(global_indices)
            
            # Take remaining indices starting from global_progress
            remaining_indices = global_indices[global_progress:]
            
            # Partition remaining work among new world size
            samples_per_rank = len(remaining_indices) // new_world_size
            remainder = len(remaining_indices) % new_world_size
            
            start_idx = new_rank * samples_per_rank + min(new_rank, remainder)
            end_idx = start_idx + samples_per_rank + (1 if new_rank < remainder else 0)
            
            new_dataset.assigned_global_indices = remaining_indices[start_idx:end_idx]
            new_dataset.local_samples_processed = 0

        logger.info(f"Created dataset for new world size {new_world_size}, "
                   f"rank {new_rank}: assigned {len(new_dataset.assigned_global_indices)} indices")

        return new_dataset

    def get_progress_info(self) -> Dict[str, Any]:
        """Get detailed progress information for monitoring"""
        return {
            'rank': self.dist_rank,
            'world_size': self.dist_world_size,
            'assigned_samples': len(self.assigned_global_indices),
            'processed_samples': self.local_samples_processed,
            'remaining_samples': len(self),
            'progress_percent': (self.local_samples_processed / 
                               len(self.assigned_global_indices) * 100) 
                              if len(self.assigned_global_indices) > 0 else 100.0,
            'assigned_indices_range': f"[{min(self.assigned_global_indices)}, {max(self.assigned_global_indices)}]" 
                                    if self.assigned_global_indices else "[]",
            'current_global_idx': self.assigned_global_indices[self.local_samples_processed - 1] 
                                if self.local_samples_processed > 0 and self.local_samples_processed <= len(self.assigned_global_indices) else None
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
