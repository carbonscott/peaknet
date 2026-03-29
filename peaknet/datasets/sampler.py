"""
Checkpointable distributed sampler with world-size-agnostic resume.

Separates sample ordering and progress tracking from data access,
allowing PeakNetDataset to be a pure data accessor.
"""

import random
import logging
from typing import Iterator, Dict, Any, List

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class CheckpointableSampler(Sampler[int]):
    """Distributed sampler with world-size-agnostic checkpoint/resume.

    Partitions ``total_samples`` global indices across ranks using
    shuffle-then-partition.  Checkpoint state is global-only (no
    rank-specific state), so training can resume with a different
    world size.

    Implements ``state_dict`` / ``load_state_dict`` so it can be
    registered directly with ``accelerator.register_for_checkpointing()``.
    """

    def __init__(
        self,
        total_samples: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = 42,
        reshuffle_frequency: int = 0,
    ):
        self.total_samples = total_samples
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.shuffle_seed_base = seed
        self.reshuffle_frequency = reshuffle_frequency

        self.current_shuffle_epoch = 0
        self.current_index = 0

        self._build_indices()

    # -- Index construction ---------------------------------------------------

    def _build_indices(self) -> None:
        """Build rank-local index list from global state."""
        if self.total_samples == 0:
            self.indices: List[int] = []
            return

        all_samples = list(range(self.total_samples))
        self.indices = self._partition(all_samples)

        logger.info(
            f"Rank {self.rank}: Assigned {len(self.indices)} global indices "
            f"(shuffling {'enabled' if self.shuffle else 'disabled'})"
        )

    def _partition(self, sample_list: List[int]) -> List[int]:
        """Shuffle-then-partition a sample list across ranks."""
        if not sample_list:
            return []

        samples = sample_list.copy()
        if self.shuffle:
            seed = self.shuffle_seed_base + self.current_shuffle_epoch
            rng = random.Random(seed)
            rng.shuffle(samples)

        per_rank = len(samples) // self.world_size
        remainder = len(samples) % self.world_size
        start = self.rank * per_rank + min(self.rank, remainder)
        end = start + per_rank + (1 if self.rank < remainder else 0)
        return samples[start:end]

    # -- Sampler interface ----------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    # -- Progress tracking ----------------------------------------------------

    def advance(self, num_samples: int) -> None:
        """Advance the progress counter with epoch cycling."""
        self.current_index += num_samples

        if len(self.indices) > 0 and self.current_index >= len(self.indices):
            epochs_completed = self.current_index // len(self.indices)
            self.current_index = self.current_index % len(self.indices)
            if epochs_completed > 0:
                logger.info(
                    f"Rank {self.rank}: Completed {epochs_completed} epoch(s), "
                    f"reset progress to index {self.current_index}"
                )

    def maybe_reshuffle(self, current_step: int) -> None:
        """Reshuffle if the step matches the reshuffle frequency."""
        if (
            self.shuffle
            and self.reshuffle_frequency > 0
            and current_step > 0
            and current_step % self.reshuffle_frequency == 0
        ):
            self.current_shuffle_epoch += 1
            self._build_indices()
            self.current_index = 0
            logger.info(
                f"Rank {self.rank}: Reshuffled at step {current_step}, "
                f"new shuffle epoch {self.current_shuffle_epoch}"
            )

    # -- Checkpointing --------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_shuffle_epoch": self.current_shuffle_epoch,
            "shuffle_seed_base": self.shuffle_seed_base,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_shuffle_epoch = state_dict.get(
            "current_shuffle_epoch", 0
        )
        self.shuffle_seed_base = state_dict.get(
            "shuffle_seed_base", self.shuffle_seed_base
        )
        self._build_indices()
        logger.info(
            f"Restored sampler state: shuffle epoch {self.current_shuffle_epoch}, "
            f"seed base {self.shuffle_seed_base}"
        )
