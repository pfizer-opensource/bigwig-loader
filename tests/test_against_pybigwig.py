import numpy as np
import pandas as pd
import pyBigWig
import pytest

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.collection import interpret_path


class PyBigWigCollection:
    def __init__(self, bigwig_path, first_n_files=None):
        bigwig_paths = sorted(interpret_path(bigwig_path))
        self.bigwigs = [pyBigWig.open(str(bw)) for bw in bigwig_paths][:first_n_files]

    def get_batch(self, chromosomes, starts, ends):
        out = np.stack(
            [
                bw.values(chrom, start, end, numpy=True)
                for chrom, start, end in zip(chromosomes, starts, ends)
                for bw in self.bigwigs
            ]
        )
        return out.reshape(
            (out.shape[0] // len(self.bigwigs), len(self.bigwigs), out.shape[1])
        )


def test_same_output(bigwig_path):
    pybigwig_collection = PyBigWigCollection(bigwig_path, first_n_files=3)
    collection = BigWigCollection(bigwig_path, first_n_files=3)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]
    chromosomes, starts, ends = (
        list(df["chr"]),
        list(df["center"] - 500),
        list(df["center"] + 500),
    )

    pybigwig_batch = pybigwig_collection.get_batch(chromosomes, starts, ends)
    np.nan_to_num(pybigwig_batch, copy=False, nan=0.0)
    this_batch = collection.get_batch(chromosomes, starts, ends).get()
    print("PyBigWig:")
    print(pybigwig_batch)
    print(type(this_batch), "shape:", pybigwig_batch.shape)
    print("This Library:")
    print(this_batch)
    print(type(this_batch), "shape:", this_batch.shape)
    print(this_batch[pybigwig_batch != this_batch])
    print(pybigwig_batch[pybigwig_batch != this_batch])
    assert (pybigwig_batch == this_batch).all()


def test_same_output_with_nans(bigwig_path):
    pybigwig_collection = PyBigWigCollection(bigwig_path, first_n_files=3)
    collection = BigWigCollection(bigwig_path, first_n_files=3)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]
    chromosomes, starts, ends = (
        list(df["chr"]),
        list(df["center"] - 1000),
        list(df["center"] + 1000),
    )

    pybigwig_batch = pybigwig_collection.get_batch(chromosomes, starts, ends)

    this_batch = collection.get_batch(
        chromosomes, starts, ends, default_value=np.nan
    ).get()
    print("PyBigWig:")
    print(pybigwig_batch)
    print(type(this_batch), "shape:", pybigwig_batch.shape)
    print("This Library:")
    print(this_batch)
    print(type(this_batch), "shape:", this_batch.shape)
    print(this_batch[pybigwig_batch != this_batch])
    print(pybigwig_batch[pybigwig_batch != this_batch])
    assert np.allclose(pybigwig_batch, this_batch, equal_nan=True)


@pytest.mark.parametrize("window_size", [2, 11, 128])
@pytest.mark.parametrize("default_value", [0.0, np.nan, 2.0, 5.6, 10])
@pytest.mark.parametrize("sequence_length", [1000, 2048])
def test_windowed_output_against_pybigwig(
    bigwig_path, window_size, default_value, sequence_length
):
    print("window_size:", window_size)
    pybigwig_collection = PyBigWigCollection(bigwig_path, first_n_files=3)
    collection = BigWigCollection(bigwig_path, first_n_files=3)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]

    chromosomes = list(df["chr"])
    starts = list(df["center"] - sequence_length // 2)
    ends = [position + sequence_length for position in starts]

    pybigwig_batch = pybigwig_collection.get_batch(chromosomes, starts, ends)

    this_batch = collection.get_batch(
        chromosomes, starts, ends, default_value=default_value, window_size=window_size
    ).get()

    # Reshape the tensor so the last dimension contains
    # all the values corresponding to one window
    reduced_dim = sequence_length // window_size
    pybigwig_batch = pybigwig_batch[:, :, : reduced_dim * window_size]
    pybigwig_batch = pybigwig_batch.reshape(
        pybigwig_batch.shape[0], pybigwig_batch.shape[1], reduced_dim, window_size
    )

    # fill nan's with the chosen default value
    pybigwig_batch = np.nan_to_num(pybigwig_batch, copy=False, nan=default_value)
    # And take mean over the window
    pybigwig_batch = np.nanmean(pybigwig_batch, axis=-1)

    print("PyBigWig (with window function applied afterwards):")
    print(pybigwig_batch)
    print("bigwig-loader:")
    print(this_batch)
    assert np.allclose(pybigwig_batch, this_batch, equal_nan=True)
