import torch
from itertools import cycle


def create_infinite_dataloader(dataset, base_seed, drop_last_in_sampler, drop_last_in_loader, 
                              uses_dist, batch_size, num_workers, pin_memory, prefetch_factor):
    """Create an infinite cycling dataloader that repeats the dataset indefinitely.
    
    Args:
        dataset: PyTorch dataset
        base_seed: Base seed for sampling
        drop_last_in_sampler: Whether to drop last incomplete batch in sampler
        drop_last_in_loader: Whether to drop last incomplete batch in loader
        uses_dist: Whether distributed training is used
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of samples loaded in advance by each worker
        
    Returns:
        tuple: (infinite_dataloader, sampler, batches_per_epoch) where sampler is None if not using distributed
    """
    # Create base dataloader
    dataloader, sampler = wrap_with_torch_dataloader(
        dataset=dataset,
        base_seed=base_seed,
        drop_last_in_sampler=drop_last_in_sampler,
        drop_last_in_loader=drop_last_in_loader,
        uses_dist=uses_dist,
        batch_size=batch_size,
        num_workers=num_workers,
        custom_collate=None,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        epoch=0,  # Fixed epoch for cycling
        is_eval=False,
    )
    
    # Calculate batches per epoch for epoch-aware logic
    batches_per_epoch = len(dataloader)
    
    # Create infinite cycling iterator
    return cycle(dataloader), sampler, batches_per_epoch


def wrap_with_torch_dataloader(
    dataset,
    base_seed,
    drop_last_in_sampler,
    drop_last_in_loader,
    uses_dist,
    batch_size,
    num_workers,
    custom_collate,
    pin_memory,
    prefetch_factor,
    epoch,
    is_eval=False,
):
    """
    Convenient wrapper for creating PyTorch DataLoaders with distributed support.

    Args:
        dataset: PyTorch dataset
        base_seed: Base seed for sampling
        drop_last_in_sampler: Whether to drop last incomplete batch in sampler
        drop_last_in_loader: Whether to drop last incomplete batch in loader
        uses_dist: Whether distributed training is used
        batch_size: Batch size
        num_workers: Number of worker processes
        custom_collate: Custom collate function (can be None)
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of samples loaded in advance by each worker
        epoch: Current epoch number
        is_eval: Whether this is for evaluation (affects shuffling)

    Returns:
        tuple: (dataloader, sampler) where sampler is None if not using distributed
    """
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=True,
        seed=base_seed,
        drop_last=drop_last_in_sampler
    ) if uses_dist else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False if is_eval else (None if uses_dist else True),  # Only shuffle if not using distributed sampler
        collate_fn=custom_collate,
        drop_last=drop_last_in_loader,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    return dataloader, sampler
