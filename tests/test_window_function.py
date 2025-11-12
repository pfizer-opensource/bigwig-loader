from math import isnan

import cupy as cp
import pandas as pd
import pytest

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.dataset import BigWigDataset


@pytest.mark.parametrize("window_size", [2, 11, 32, 128])
@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_same_output(bigwig_path, window_size, default_value):
    collection = BigWigCollection(bigwig_path, first_n_files=3)
    print(collection.bigwig_paths)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]
    chromosomes, starts, ends = (
        list(df["chr"]),
        list(df["center"] - 1024),
        list(df["center"] + 1024),
    )
    full_batch = collection.get_batch(
        chromosomes, starts, ends, window_size=1, default_value=default_value
    )
    batch_with_window = collection.get_batch(
        chromosomes, starts, ends, window_size=window_size, default_value=default_value
    )
    sequence_length = 2048
    reduced_dim = sequence_length // window_size

    if isnan(default_value):
        assert cp.isnan(full_batch).any()
    else:
        assert not cp.isnan(full_batch).any()

    full_matrix = full_batch[:, : reduced_dim * window_size, :]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], reduced_dim, window_size, full_matrix.shape[-1]
    )
    expected = cp.nanmean(full_matrix, axis=-2)
    print(batch_with_window)
    print(expected)
    assert cp.allclose(expected, batch_with_window, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan, 4.0, 5.6])
def test_dataset_with_window_function(
    default_value, bigwig_path, reference_genome_path, merged_intervals
):
    center_bin_to_predict = 2048
    window_size = 128
    reduced_dim = center_bin_to_predict // window_size

    batch_size = 16

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin({"chr1", "chr3", "chr5"})]
    chromosomes = list(df["chr"])[:batch_size]
    centers = list(df["center"])[:batch_size]

    position_sampler = [(chrom, center) for chrom, center in zip(chromosomes, centers)]

    dataset = BigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=center_bin_to_predict * 2,
        center_bin_to_predict=center_bin_to_predict,
        window_size=1,
        batch_size=batch_size,
        batches_per_epoch=1,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=3,
        custom_position_sampler=position_sampler,
        default_value=default_value,
        return_batch_objects=True,
    )

    dataset_with_window = BigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=center_bin_to_predict * 2,
        center_bin_to_predict=center_bin_to_predict,
        window_size=window_size,
        batch_size=batch_size,
        batches_per_epoch=1,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=3,
        custom_position_sampler=position_sampler,
        default_value=default_value,
        return_batch_objects=True,
    )

    for batch, batch_with_window in zip(dataset, dataset_with_window):
        print(batch)
        print(batch_with_window)
        print(batch.chromosomes)
        print(batch_with_window.chromosomes)
        print(batch.starts)
        print(batch_with_window.starts)
        print(batch.ends)
        print(batch_with_window.ends)
        expected = batch.values.reshape(
            batch.values.shape[0],
            reduced_dim,
            window_size,
            batch.values.shape[-1],
        )
        if not isnan(default_value) or default_value == 0:
            cp.nan_to_num(expected, copy=False, nan=default_value)
        expected = cp.nanmean(expected, axis=-2)
        print("---")
        print("expected", expected.shape)
        print(expected)
        print("batch_with_window", batch_with_window.values.shape)
        print(batch_with_window.values)
        assert cp.allclose(expected, batch_with_window.values, equal_nan=True)
        if isnan(default_value):
            assert cp.isnan(batch_with_window.values).any()
