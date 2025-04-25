import cupy as cp
import pandas as pd
import pytest

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_same_output(bigwig_path, default_value):
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
        chromosomes, starts, ends, window_size=128, default_value=default_value
    )
    sequence_length = 2048
    window_size = 128
    reduced_dim = sequence_length // window_size
    full_matrix = full_batch[:, :, : reduced_dim * window_size]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], full_matrix.shape[1], reduced_dim, window_size
    )
    expected = cp.nanmean(full_matrix, axis=-1)
    print(batch_with_window)
    print(expected)
    assert cp.allclose(expected, batch_with_window, equal_nan=True)
