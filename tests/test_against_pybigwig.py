import numpy as np
import pandas as pd
import pyBigWig

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
    pybigwig_collection = PyBigWigCollection(bigwig_path, first_n_files=2)
    collection = BigWigCollection(bigwig_path, first_n_files=2)

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
