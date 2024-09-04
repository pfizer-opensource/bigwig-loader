import cupy as cp
import pandas as pd
import pytest

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.dataset import BigWigDataset
from bigwig_loader.path import interpret_path
from bigwig_loader.sampler.position_sampler import RandomPositionSampler
from bigwig_loader.util import sort_intervals


def _get_example_batch_of_intervals(n_bases=1000, batch_size=256):
    batch_of_positions = pd.read_csv(
        config.example_data_dir / "some_positions.tsv", sep="\t"
    )

    batch_of_positions = batch_of_positions.sample(batch_size)

    center = batch_of_positions["center"]
    batch_of_intervals = pd.DataFrame(
        {"chrom": batch_of_positions["chr"], "start": center - (n_bases // 2)}
    )
    batch_of_intervals["end"] = batch_of_intervals["start"] + n_bases
    return batch_of_intervals


@pytest.fixture
def bigwig_dataset(bigwig_path, reference_genome_path, merged_intervals):
    dataset = BigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=1000,
        center_bin_to_predict=1000,
        batch_size=256,
        batches_per_epoch=10,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=2,
    )
    return dataset


def test_dataset(bigwig_dataset):
    for sequence, target in bigwig_dataset:
        assert target.shape == (256, 2, 1000)


def test_get_batch(collection):
    intervals = _get_example_batch_of_intervals(n_bases=1000, batch_size=256)
    n_files = len(collection.bigwig_paths)
    batch = collection.get_batch(
        intervals["chrom"].values,
        intervals["start"].values,
        intervals["end"].values,
    )
    assert batch.shape == (256, n_files, 1000)


def test_exclude_intervals(collection):
    intervals = collection.intervals(
        ["chr3", "chr4", "chr5"], exclude_chromosomes=["chr4"]
    )
    present_chromosomes = set(intervals["chrom"])
    assert "chr5" in present_chromosomes and "chr4" not in present_chromosomes


def test_merge_intervals(collection, merged_intervals):
    unmerged_intervals = collection.intervals(
        ["chr3", "chr4", "chr5"], exclude_chromosomes=["chr4"], merge=False, threshold=2
    )
    assert len(merged_intervals) < len(unmerged_intervals)


def test_sort_intervals(example_data_dir):
    intervals = pd.read_csv(example_data_dir / "some_intervals.tsv", sep="\t")
    intervals.rename(columns={"chr": "chrom"}, inplace=True)
    df = sort_intervals(intervals)
    df.reset_index(drop=True, inplace=True)
    assert df[df["chrom"] == "chr2"].index[0] < df[df["chrom"] == "chr10"].index[0]


def test_position_sampler(merged_intervals):
    sampler = RandomPositionSampler(merged_intervals)
    sample = next(sampler)
    assert sample[0].startswith("chr")


def test_chromosomes_present_in_all_files(collection: BigWigCollection):
    chromosomes = collection.get_chromosomes_present_in_all_files("standard")
    assert len(chromosomes) > 0
    assert all(isinstance(c, str) and c.strip() for c in chromosomes)


def test_scaling(bigwig_path, reference_genome_path):
    stems = [path.stem for path in sorted(interpret_path(bigwig_path))]
    scale = {stems[0]: 3, stems[1]: 10}

    unscaled = BigWigCollection(
        bigwig_path,
        first_n_files=2,
    )
    scaled = BigWigCollection(
        bigwig_path,
        first_n_files=2,
        scale=scale,
    )

    intervals = _get_example_batch_of_intervals(n_bases=1000, batch_size=32)

    unscaled_batch = unscaled.get_batch(
        intervals["chrom"].values,
        intervals["start"].values,
        intervals["end"].values,
    )

    scaled_batch = scaled.get_batch(
        intervals["chrom"].values,
        intervals["start"].values,
        intervals["end"].values,
    )

    assert cp.allclose(
        scaled_batch, unscaled_batch * cp.asarray([3, 10]).reshape(1, 2, 1)
    )
