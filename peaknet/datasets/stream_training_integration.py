"""
Integration utilities for CheckpointableStreamDataset with training loops.

This module provides helper functions to replace the segment-based training
logic with the new global indexing approach.
"""

import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional, List, Tuple
from .checkpointable_stream_dataset import CheckpointableStreamDataset, CheckpointableStreamDatasetConfig
from .checkpoint_manager import CheckpointManager, create_dataloader_with_checkpoint_support

logger = logging.getLogger(__name__)


def create_stream_datasets(path_dataset_train: str, path_dataset_eval: str,
                          transforms: List, buffer_size: int,
                          dist_rank: int, dist_world_size: int, device: str,
                          dtype: Optional[torch.dtype] = None,
                          uses_norm: bool = True, scales_variance: bool = True,
                          perfs_runtime: bool = False,
                          global_index_cache: Optional[str] = None) -> Dict[str, CheckpointableStreamDataset]:
    """
    Create CheckpointableStreamDataset instances to replace SegmentedPeakNetDataset.
    
    Args:
        path_dataset_train: Path to training dataset CSV
        path_dataset_eval: Path to evaluation dataset CSV  
        transforms: List of transformations to apply
        buffer_size: Number of zarr files to cache in memory
        dist_rank: Current distributed rank
        dist_world_size: Total number of distributed ranks
        device: Device string
        dtype: Data type for tensors
        uses_norm: Whether to apply instance normalization
        scales_variance: Whether to scale variance in normalization
        perfs_runtime: Whether to measure performance
        global_index_cache: Path to global index cache file
        
    Returns:
        Dict containing 'train', 'eval_train', and 'eval_val' datasets
    """
    
    # Create training dataset config
    train_config = CheckpointableStreamDatasetConfig(
        path_csv=path_dataset_train,
        transforms=transforms,
        buffer_size=buffer_size,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        device=device,
        dtype=dtype,
        uses_norm=uses_norm,
        scales_variance=scales_variance,
        perfs_runtime=perfs_runtime,
        global_index_cache=global_index_cache
    )
    
    # Create evaluation dataset config (same as training for eval_train)
    eval_config = CheckpointableStreamDatasetConfig(
        path_csv=path_dataset_eval,
        transforms=transforms,
        buffer_size=1,  # Smaller cache for eval
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        device=device,
        dtype=dtype,
        uses_norm=uses_norm,
        scales_variance=scales_variance,
        perfs_runtime=perfs_runtime,
        global_index_cache=None  # Don't cache eval index
    )
    
    # Create dataset instances
    dataset_train = CheckpointableStreamDataset(train_config)
    dataset_eval_train = CheckpointableStreamDataset(train_config)  # Use training data for eval
    dataset_eval_val = CheckpointableStreamDataset(eval_config)
    
    logger.info(f"Created stream datasets - Train: {len(dataset_train)} samples, "
               f"Eval: {len(dataset_eval_val)} samples")
    
    return {
        'train': dataset_train,
        'eval_train': dataset_eval_train,
        'eval_val': dataset_eval_val
    }


def create_stream_dataloaders(datasets: Dict[str, CheckpointableStreamDataset],
                             batch_size: int, num_workers: int = 4,
                             pin_memory: bool = True, prefetch_factor: int = 4) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create optimized DataLoaders for the stream datasets.
    
    Args:
        datasets: Dict of CheckpointableStreamDataset instances
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Use pinned memory for fast H2D transfer
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        Dict containing DataLoaders for each dataset
    """
    
    dataloaders = {}
    
    for key, dataset in datasets.items():
        if len(dataset) == 0:
            logger.warning(f"Dataset '{key}' has no samples for rank {dataset.dist_rank}")
            dataloaders[key] = None
            continue
            
        dataloader = create_dataloader_with_checkpoint_support(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )
        
        dataloaders[key] = dataloader
        logger.info(f"Created DataLoader for '{key}': {len(dataloader)} batches")
    
    return dataloaders


class StreamTrainingCoordinator:
    """
    Coordinates training loop with CheckpointableStreamDataset and checkpoint management.
    
    Replaces the segment-based training logic with a simpler epoch-based approach.
    """
    
    def __init__(self, datasets: Dict[str, CheckpointableStreamDataset],
                 dist_rank: int, dist_world_size: int):
        self.datasets = datasets
        self.dist_rank = dist_rank
        self.dist_world_size = dist_world_size
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(dist_rank, dist_world_size)
        
        # Track training progress
        self.current_epoch = 0
        self.global_step = 0
        
    def prepare_epoch(self, epoch: int, from_resume: bool = False) -> bool:
        """
        Prepare datasets for new epoch.
        
        Args:
            epoch: Current epoch number
            from_resume: Whether this is resuming from checkpoint
            
        Returns:
            True if epoch preparation successful, False if no data remaining
        """
        self.current_epoch = epoch
        
        if not from_resume:
            # Reset progress for new epoch
            for dataset in self.datasets.values():
                if dataset is not None:
                    dataset.reset_progress()
        
        # Check if training dataset has remaining samples
        train_dataset = self.datasets.get('train')
        if train_dataset is None or len(train_dataset) == 0:
            logger.info(f"No training data remaining for rank {self.dist_rank}")
            return False
        
        logger.info(f"Epoch {epoch}: Rank {self.dist_rank} has {len(train_dataset)} training samples")
        return True
    
    def create_train_dataloader(self, batch_size: int, num_workers: int = 4,
                               pin_memory: bool = True, prefetch_factor: int = 4) -> Optional[torch.utils.data.DataLoader]:
        """
        Create training DataLoader for current epoch.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Use pinned memory
            prefetch_factor: Prefetch factor
            
        Returns:
            DataLoader or None if no data available
        """
        train_dataset = self.datasets.get('train')
        if train_dataset is None or len(train_dataset) == 0:
            return None
        
        return create_dataloader_with_checkpoint_support(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )
    
    def update_training_progress(self, num_batches_processed: int, batch_size: int):
        """
        Update training progress after processing batches.
        
        Args:
            num_batches_processed: Number of batches processed in this iteration
            batch_size: Batch size
        """
        train_dataset = self.datasets.get('train')
        if train_dataset is not None:
            samples_processed = num_batches_processed * batch_size
            train_dataset.advance_local_progress(samples_processed)
            self.global_step += num_batches_processed
    
    def get_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """
        Get global checkpoint state.
        
        Returns:
            Global checkpoint state (only on rank 0)
        """
        train_dataset = self.datasets.get('train')
        if train_dataset is None:
            return None
        
        local_state = train_dataset.get_checkpoint_state()
        return self.checkpoint_manager.aggregate_global_progress(local_state)
    
    def save_checkpoint(self, checkpoint_path: str, model_state: Dict, optimizer_state: Dict,
                       additional_state: Optional[Dict] = None):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            additional_state: Additional state to save
        """
        global_state = self.get_checkpoint_state()
        if global_state is not None:  # Only rank 0
            additional_state = additional_state or {}
            additional_state.update({
                'current_epoch': self.current_epoch,
                'global_step': self.global_step
            })
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint_path=checkpoint_path,
                global_state=global_state,
                model_state=model_state,
                optimizer_state=optimizer_state,
                additional_state=additional_state
            )
    
    def load_and_resume(self, checkpoint_path: str, new_world_size: Optional[int] = None,
                       new_rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            new_world_size: New world size (if different from original)
            new_rank: New rank (if different from original)
            
        Returns:
            Loaded checkpoint data
        """
        # Use current values if not specified
        if new_world_size is None:
            new_world_size = self.dist_world_size
        if new_rank is None:
            new_rank = self.dist_rank
        
        # Load checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Validate resumption
        train_dataset = self.datasets.get('train')
        if train_dataset is not None:
            file_paths = train_dataset.file_paths
            if not self.checkpoint_manager.validate_resumption(checkpoint_data, file_paths):
                raise ValueError("Checkpoint validation failed")
        
        # Handle world size change
        if new_world_size != self.dist_world_size or new_rank != self.dist_rank:
            logger.info(f"Resuming with different configuration: "
                       f"{self.dist_world_size}→{new_world_size} GPUs, "
                       f"rank {self.dist_rank}→{new_rank}")
            
            # Create new dataset instances for changed world size
            self._recreate_datasets_for_resumption(checkpoint_data, new_world_size, new_rank)
        else:
            # Same world size, just restore state
            for dataset in self.datasets.values():
                if dataset is not None:
                    dataset_state = checkpoint_data['dataset_state']['rank_states'][self.dist_rank]
                    dataset.restore_checkpoint_state(dataset_state)
        
        # Restore coordinator state
        additional_state = checkpoint_data.get('additional_state', {})
        self.current_epoch = additional_state.get('current_epoch', 0)
        self.global_step = additional_state.get('global_step', 0)
        
        return checkpoint_data
    
    def _recreate_datasets_for_resumption(self, checkpoint_data: Dict[str, Any],
                                         new_world_size: int, new_rank: int):
        """Recreate datasets for resumption with different world size"""
        dataset_state = checkpoint_data['dataset_state']
        global_progress = dataset_state['global_samples_processed']
        
        # Recreate training dataset
        train_dataset = self.datasets.get('train')
        if train_dataset is not None:
            new_train_dataset = train_dataset.create_for_new_world_size(
                new_world_size, new_rank, global_progress
            )
            self.datasets['train'] = new_train_dataset
        
        # Recreate eval datasets (they get fresh assignments)
        for key in ['eval_train', 'eval_val']:
            eval_dataset = self.datasets.get(key)
            if eval_dataset is not None:
                new_eval_dataset = eval_dataset.create_for_new_world_size(
                    new_world_size, new_rank, 0  # Eval starts fresh
                )
                self.datasets[key] = new_eval_dataset
        
        # Update coordinator state
        self.dist_world_size = new_world_size
        self.dist_rank = new_rank
        self.checkpoint_manager = CheckpointManager(new_rank, new_world_size)
    
    def get_training_progress_summary(self) -> Dict[str, Any]:
        """Get summary of training progress for monitoring"""
        summaries = {}
        
        for key, dataset in self.datasets.items():
            if dataset is not None:
                summaries[key] = dataset.get_progress_info()
        
        return {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'datasets': summaries
        }
    
    def create_eval_dataloader(self, dataset_key: str, batch_size: int,
                              subset_size: Optional[int] = None) -> Optional[torch.utils.data.DataLoader]:
        """
        Create evaluation DataLoader, optionally with subset sampling.
        
        Args:
            dataset_key: 'eval_train' or 'eval_val'
            batch_size: Batch size for evaluation
            subset_size: If specified, randomly sample this many samples for evaluation
            
        Returns:
            DataLoader for evaluation or None if no data
        """
        eval_dataset = self.datasets.get(dataset_key)
        if eval_dataset is None or len(eval_dataset) == 0:
            logger.warning(f"No data available for {dataset_key}")
            return None
        
        # For subset evaluation, we could implement a SubsetSampler
        # For now, just return full dataset
        return create_dataloader_with_checkpoint_support(
            dataset=eval_dataset,
            batch_size=batch_size,
            num_workers=2,  # Fewer workers for eval
            pin_memory=True,
            persistent_workers=False,  # Don't keep eval workers persistent
            prefetch_factor=2
        )