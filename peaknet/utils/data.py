import torch


def _infinite_iter(dataloader):
    """Yield batches from *dataloader* forever, re-invoking its iterator
    each time it is exhausted so the sampler is re-queried every epoch."""
    while True:
        yield from dataloader


def create_infinite_dataloader(dataset, sampler, drop_last_in_loader,
                               batch_size, num_workers, custom_collate,
                               pin_memory, prefetch_factor):
    """Create an infinite dataloader that re-queries its sampler each epoch.

    Args:
        dataset: PyTorch dataset (pure data accessor)
        sampler: A sampler (e.g. CheckpointableSampler) that yields indices
        drop_last_in_loader: Whether to drop last incomplete batch in loader
        batch_size: Batch size
        num_workers: Number of worker processes
        custom_collate: Custom collate function (can be None)
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of samples loaded in advance by each worker

    Returns:
        tuple: (infinite_dataloader, batches_per_epoch)
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=drop_last_in_loader,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    batches_per_epoch = len(dataloader)
    return _infinite_iter(dataloader), batches_per_epoch


def wrap_with_torch_dataloader(
    dataset,
    sampler,
    drop_last_in_loader,
    batch_size,
    num_workers,
    custom_collate,
    pin_memory,
    prefetch_factor,
):
    """
    Create a PyTorch DataLoader with an external sampler.

    Args:
        dataset: PyTorch dataset (pure data accessor)
        sampler: A sampler that yields indices for this rank
        drop_last_in_loader: Whether to drop last incomplete batch in loader
        batch_size: Batch size
        num_workers: Number of worker processes
        custom_collate: Custom collate function (can be None)
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of samples loaded in advance by each worker

    Returns:
        DataLoader instance
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=drop_last_in_loader,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    return dataloader
