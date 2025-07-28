import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import os

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages distributed checkpoint state for CheckpointableStreamDataset.

    Handles:
    - Aggregating progress from all ranks
    - Saving/loading global checkpoint state
    - Coordinating resumption with variable GPU counts
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.is_main_rank = (rank == 0)

    def aggregate_global_progress(self, local_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Aggregate checkpoint state from all ranks.

        Args:
            local_state: Local checkpoint state from dataset

        Returns:
            Global checkpoint state (only on rank 0, None on other ranks)
        """
        if not dist.is_initialized():
            # Non-distributed case
            global_progress = local_state['global_samples_processed']
            total_samples = local_state['total_global_samples']
            return {
                'global_samples_processed': global_progress,
                'total_global_samples': total_samples,
                'original_world_size': self.world_size,
                'rank_states': [local_state],
                'progress_percent': (global_progress / total_samples * 100) if total_samples > 0 else 100.0
            }

        # Gather all local states on rank 0
        gathered_states = [None] * self.world_size
        dist.all_gather_object(gathered_states, local_state)

        if self.is_main_rank:
            # Calculate global progress
            total_global_progress = sum(state['global_samples_processed'] for state in gathered_states)
            total_samples = gathered_states[0]['total_global_samples']  # Should be same for all ranks

            # Validate consistency
            for i, state in enumerate(gathered_states):
                if state['total_global_samples'] != total_samples:
                    logger.warning(f"Rank {i} has different total_samples: "
                                 f"{state['total_global_samples']} vs {total_samples}")
                if state['dist_world_size'] != self.world_size:
                    logger.warning(f"Rank {i} has different world_size: "
                                 f"{state['dist_world_size']} vs {self.world_size}")

            global_state = {
                'global_samples_processed': total_global_progress,
                'total_global_samples': total_samples,
                'original_world_size': self.world_size,
                'rank_states': gathered_states,
                'progress_percent': (total_global_progress / total_samples * 100) if total_samples > 0 else 100.0
            }

            logger.info(f"Global progress: {total_global_progress}/{total_samples} "
                       f"({global_state['progress_percent']:.2f}%)")

            return global_state

        return None

    def save_checkpoint(self, checkpoint_path: str, global_state: Dict[str, Any], 
                       model_state: Optional[Dict] = None, 
                       optimizer_state: Optional[Dict] = None,
                       additional_state: Optional[Dict] = None):
        """
        Save global checkpoint state.

        Args:
            checkpoint_path: Path to save checkpoint
            global_state: Global dataset state from aggregate_global_progress
            model_state: Model state dict (optional)
            optimizer_state: Optimizer state dict (optional)
            additional_state: Any additional state to save (optional)
        """
        if not self.is_main_rank:
            return

        checkpoint_data = {
            'dataset_state': global_state,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'additional_state': additional_state or {},
            'checkpoint_version': '1.0'
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint state.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data dict
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            if 'dataset_state' not in checkpoint_data:
                raise ValueError("Invalid checkpoint: missing dataset_state")

            dataset_state = checkpoint_data['dataset_state']
            logger.info(f"Loaded checkpoint: {dataset_state['global_samples_processed']}"
                       f"/{dataset_state['total_global_samples']} samples processed")

            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def coordinate_resumption(self, checkpoint_data: Dict[str, Any], 
                            new_world_size: int, new_rank: int) -> Dict[str, Any]:
        """
        Coordinate resumption with potentially different world size.

        Args:
            checkpoint_data: Loaded checkpoint data
            new_world_size: New world size for resumption
            new_rank: New rank for this process

        Returns:
            Resumption configuration for this rank
        """
        dataset_state = checkpoint_data['dataset_state']
        global_progress = dataset_state['global_samples_processed']
        total_samples = dataset_state['total_global_samples']
        original_world_size = dataset_state['original_world_size']

        logger.info(f"Coordinating resumption: {original_world_size} → {new_world_size} GPUs")
        logger.info(f"Global progress: {global_progress}/{total_samples}")

        # Calculate remaining work
        remaining_samples = total_samples - global_progress

        if remaining_samples <= 0:
            logger.info("All samples already processed")
            return {
                'global_progress': global_progress,
                'remaining_samples': 0,
                'new_rank_range': (total_samples, total_samples),
                'resumption_successful': True
            }

        # Distribute remaining work across new ranks
        samples_per_rank = remaining_samples // new_world_size
        remainder = remaining_samples % new_world_size

        # Calculate this rank's range
        if new_rank < remainder:
            rank_start_offset = new_rank * (samples_per_rank + 1)
            rank_samples = samples_per_rank + 1
        else:
            rank_start_offset = remainder * (samples_per_rank + 1) + (new_rank - remainder) * samples_per_rank
            rank_samples = samples_per_rank

        # Convert to global indices
        new_start_idx = global_progress + rank_start_offset
        new_end_idx = new_start_idx + rank_samples

        resumption_config = {
            'global_progress': global_progress,
            'remaining_samples': remaining_samples,
            'new_rank_range': (new_start_idx, new_end_idx),
            'resumption_successful': True,
            'samples_for_this_rank': rank_samples
        }

        logger.info(f"Rank {new_rank}: Assigned {rank_samples} samples, "
                   f"global range [{new_start_idx}, {new_end_idx})")

        return resumption_config

    def validate_resumption(self, checkpoint_data: Dict[str, Any], file_paths: List[str]) -> bool:
        """
        Validate that resumption is possible with current dataset.

        Args:
            checkpoint_data: Loaded checkpoint data
            file_paths: Current dataset file paths

        Returns:
            True if resumption is valid
        """
        try:
            dataset_state = checkpoint_data['dataset_state']

            # Check if total samples is consistent
            # Note: We can't check file paths directly since GlobalIndexManager
            # will handle validation when it's created

            total_samples = dataset_state['total_global_samples']
            global_progress = dataset_state['global_samples_processed']

            if global_progress > total_samples:
                logger.error(f"Invalid checkpoint: progress {global_progress} > total {total_samples}")
                return False

            if global_progress < 0:
                logger.error(f"Invalid checkpoint: negative progress {global_progress}")
                return False

            logger.info("Checkpoint validation passed")
            return True

        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False

    def get_resumption_summary(self, checkpoint_data: Dict[str, Any], 
                              new_world_size: int) -> Dict[str, Any]:
        """
        Get summary of resumption plan without actually performing it.

        Args:
            checkpoint_data: Loaded checkpoint data
            new_world_size: Planned new world size

        Returns:
            Resumption summary
        """
        dataset_state = checkpoint_data['dataset_state']
        global_progress = dataset_state['global_samples_processed']
        total_samples = dataset_state['total_global_samples']
        original_world_size = dataset_state['original_world_size']

        remaining_samples = total_samples - global_progress
        samples_per_rank = remaining_samples // new_world_size if new_world_size > 0 else 0
        remainder = remaining_samples % new_world_size if new_world_size > 0 else 0

        # Calculate load distribution
        rank_loads = []
        for rank in range(new_world_size):
            if rank < remainder:
                rank_samples = samples_per_rank + 1
            else:
                rank_samples = samples_per_rank
            rank_loads.append(rank_samples)

        return {
            'original_world_size': original_world_size,
            'new_world_size': new_world_size,
            'total_samples': total_samples,
            'completed_samples': global_progress,
            'remaining_samples': remaining_samples,
            'progress_percent': (global_progress / total_samples * 100) if total_samples > 0 else 100.0,
            'samples_per_rank': rank_loads,
            'min_samples_per_rank': min(rank_loads) if rank_loads else 0,
            'max_samples_per_rank': max(rank_loads) if rank_loads else 0,
            'load_balance_ratio': (min(rank_loads) / max(rank_loads)) if rank_loads and max(rank_loads) > 0 else 1.0
        }


def create_dataloader_with_checkpoint_support(dataset, batch_size: int, num_workers: int = 4,
                                             pin_memory: bool = True, persistent_workers: bool = True,
                                             prefetch_factor: int = 4) -> torch.utils.data.DataLoader:
    """
    Create optimized DataLoader with checkpoint-compatible settings.

    Args:
        dataset: CheckpointableStreamDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Use pinned memory for fast H2D transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        Configured DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,  # Dataset handles sample distribution
        drop_last=True,  # For consistent batch sizes across ranks
        collate_fn=None  # Use default collation
    )
