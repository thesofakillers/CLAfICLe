import datasets


def yield_batches_from_stream(
    dataset: datasets.IterableDataset, column_name: str, batch_size: int = 1000
):
    """yields batches of size batch_size from an iterable dataset"""
    for batch in dataset.iter(batch_size=batch_size):
        yield batch[column_name]
