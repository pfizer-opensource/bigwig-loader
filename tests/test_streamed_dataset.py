import random
from typing import Generator

import pytest

from bigwig_loader.batch import Batch
from bigwig_loader.streamed_dataset import StreamedDataloader


@pytest.mark.timeout(10)
def test_dataloader(collection, example_intervals):
    all_chrom, all_start, all_end = example_intervals
    super_batch_size = 32
    batch_size = 5

    sequence_length = all_end[0] - all_start[0]

    # setting random seed to make reproducible results
    # but should obviously work with any random seed
    sample = random.Random(56).sample

    def input_generator() -> (
        Generator[tuple[list[str], list[int], list[int]], None, None]
    ):
        while True:
            random_indices = sample(range(len(all_start)), super_batch_size)
            chrom = [all_chrom[i] for i in random_indices]
            start = [all_start[i] for i in random_indices]
            end = [all_end[i] for i in random_indices]
            yield (chrom, start, end)

    data_loader = StreamedDataloader(
        input_generator=input_generator(),
        collection=collection,
        num_threads=4,
        queue_size=4,
        slice_size=batch_size,
    )

    for i, batch in enumerate(data_loader):
        if i == 2:
            break
    data_loader.stop()

    assert batch.values.shape == (batch_size, len(collection), sequence_length)
    assert len(batch.starts) == batch_size
    assert len(batch.ends) == batch_size
    assert len(batch.chromosomes) == batch_size


@pytest.mark.timeout(10)
def test_dataloader_with_sequences_correct_slicing(collection, example_intervals):
    all_chrom, all_start, all_end = example_intervals
    super_batch_size = 32
    batch_size = 5

    sequence_length = all_end[0] - all_start[0]

    # setting random seed to make reproducible results
    # but should obviously work with any random seed
    sample = random.Random(56).sample

    def input_generator() -> (
        Generator[tuple[list[str], list[int], list[int]], None, None]
    ):
        i = 0
        while True:
            random_indices = sample(range(len(all_start)), super_batch_size)
            chrom = [all_chrom[i] for i in random_indices]
            start = [all_start[i] for i in random_indices]
            end = [all_end[i] for i in random_indices]
            # There is nothing special about the sequences, they could be anything
            yield Batch(
                chrom,
                start,
                end,
                sequences=[
                    (i, batch_index, item_index)
                    for batch_index in range(super_batch_size // batch_size)
                    for item_index in range(batch_size)
                ],
            )
            i += 1

    data_loader = StreamedDataloader(
        input_generator=input_generator(),
        collection=collection,
        num_threads=4,
        queue_size=4,
        slice_size=batch_size,
    )

    for i, batch in enumerate(data_loader):
        assert batch.values.shape == (batch_size, len(collection), sequence_length)
        assert len(batch.starts) == batch_size
        assert len(batch.ends) == batch_size
        assert len(batch.chromosomes) == batch_size
        assert len(batch.sequences) == batch_size

        super_batch_indices = [seq[0] for seq in batch.sequences]
        batch_indices = [seq[1] for seq in batch.sequences]
        item_indices = [seq[2] for seq in batch.sequences]

        assert len(set(super_batch_indices)) == 1
        assert len(set(batch_indices)) == 1
        assert item_indices == [i for i in range(batch_size)]

        if i == 10:
            break
    data_loader.stop()


@pytest.mark.timeout(10)
def test_dataloader_with_track_indices(collection, example_intervals):
    all_chrom, all_start, all_end = example_intervals
    super_batch_size = 32
    batch_size = 5

    sequence_length = all_end[0] - all_start[0]

    # setting random seed to make reproducible results
    # but should obviously work with any random seed
    sample = random.Random(56).sample

    def input_generator() -> (
        Generator[tuple[list[str], list[int], list[int]], None, None]
    ):
        i = 0
        while True:
            random_indices = sample(range(len(all_start)), super_batch_size)
            chrom = [all_chrom[i] for i in random_indices]
            start = [all_start[i] for i in random_indices]
            end = [all_end[i] for i in random_indices]
            # There is nothing special about the sequences, they could be anything
            if i % 4 == 0:
                track_indices = [0]
            elif i % 4 == 1:
                track_indices = [1]
            elif i % 4 == 2:
                track_indices = [0, 1]
            else:
                track_indices = [1, 0]
            yield Batch(
                chrom,
                start,
                end,
                track_indices=track_indices,
                other={"n_tracks": len(track_indices)},
            )
            i += 1

    data_loader = StreamedDataloader(
        input_generator=input_generator(),
        collection=collection,
        num_threads=4,
        queue_size=4,
        slice_size=batch_size,
    )

    for i, batch in enumerate(data_loader):
        assert batch.values.shape == (
            batch_size,
            batch.other["n_tracks"],
            sequence_length,
        )
        assert len(batch.starts) == batch_size
        assert len(batch.ends) == batch_size
        assert len(batch.chromosomes) == batch_size

        if i == 10:
            break
    data_loader.stop()


if __name__ == "__main__":
    import pandas as pd

    from bigwig_loader import config
    from bigwig_loader.collection import BigWigCollection
    from bigwig_loader.download_example_data import get_example_bigwigs_files

    bigwig_path = get_example_bigwigs_files()
    collection = BigWigCollection(bigwig_path, first_n_files=2)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]
    example_intervals = (
        list(df["chr"]),
        list(df["center"] - 500),
        list(df["center"] + 500),
    )

    test_dataloader(collection, example_intervals)
