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
    Configuration for PeakNetDataset (pure data accessor).

    Attributes:
        path_csv (str): CSV file to configure the dataset list
        transforms (List): A list of transformations to apply to each data item
        buffer_size (int): The number of zarr files buffered in memory (LRU cache)
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
    device: str
    dtype: Optional[torch.dtype] = None
    uses_norm: bool = True
    scales_variance: bool = True
    perfs_runtime: bool = False
    global_index_cache: Optional[str] = None


class PeakNetDataset(Dataset):
    """
    Pure data accessor for PeakNet zarr datasets.

    Owns only data loading, caching, transforms, and normalization.
    Distributed sampling and progress tracking live in
    ``CheckpointableSampler`` (see ``peaknet.datasets.sampler``).
    """

    def __init__(self, config: PeakNetDatasetConfig):
        super().__init__()

        self.config = config
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

        # Zarr file caching
        self.zarr_cache = OrderedDict()

        logger.info(f"PeakNetDataset: {self.global_manager.total_samples} total samples")

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
        """Return total number of samples in the dataset."""
        return self.global_manager.total_samples

    def __getitem__(self, global_idx: int):
        """
        Get sample by global index.

        Args:
            global_idx: Global sample index (0 to total_samples-1).
                        The sampler is responsible for choosing which indices
                        to pass here.

        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Index {global_idx} out of range [0, {len(self)})")

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

    def validate_dataset(self) -> bool:
        """Validate dataset configuration and global index."""
        try:
            if not self.global_manager.validate_index():
                return False

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

            logger.info("Dataset validation passed")
            return True

        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
