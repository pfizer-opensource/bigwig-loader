"""
This benchmarks the scenario of using pyBigWig
in a pytorch dataloader with differen number
of workers.

0 32 0.24952834727099307 0.009142287326556482
1 32 0.2645324201394732 0.15919867478319427
2 32 0.1419987794584953 0.1940197516454249
4 32 0.08257061717315362 0.21923183762619575
8 32 0.052774008721686326 0.2243132436523036
16 32 0.038562032580375674 0.25572177004054747
"""

from torch.multiprocessing import set_start_method

try:
    set_start_method("spawn", force=True)
    print("spawned")
except RuntimeError:
    pass

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyBigWig
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from bigwig_loader import config


def get_bigwig_files_from_path(
    path: Path, file_extensions: Iterable[str] = (".bigWig", ".bw"), crawl: bool = True
) -> list[Path]:
    """
    For a path object, get all bigwig files. If the path is directly pointing
    to a bigwig file, a list with just that file is returned. In case of a
    directory all files are gathered with file extensions that are part of
    file_extensions.
    Args:
        path: or upath.Path object. Either directory or BigWig file
        file_extensions: used to find all BigWig files in a directory.
            Default: (".bigWig", ".bw")
        crawl: whether to find BigWig files in subdirectories. Default: True.

    Returns: list of paths to BigWig files.

    """
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")
    elif path.is_dir():
        if crawl:
            pattern = "**/*"
        else:
            pattern = "*"
        return [
            file
            for extension in file_extensions
            for file in path.glob(f"{pattern}{extension}")
        ]
    return [path]


def interpret_path(
    bigwig_path,
    file_extensions: Iterable[str] = (".bigWig", ".bw"),
    crawl: bool = True,
) -> list[Path]:
    """
    Get all bigwig files for a path. Also excepts strings.
    If the path is directly pointing to a bigwig file, a list with just that
    file is returned. In case of a directory all files are gathered with file
    extensions that are part of file_extensions.
    Args:
        bigwig_path: str, Path or upath.Path object. Either directory or BigWig file
        file_extensions: used to find all BigWig files in a directory.
            Default: (".bigWig", ".bw")
        crawl: whether to find BigWig files in subdirectories. Default: True.

    Returns: list of paths to BigWig files.

    """
    if isinstance(bigwig_path, str) or isinstance(bigwig_path, os.PathLike):
        return get_bigwig_files_from_path(
            Path(bigwig_path), file_extensions=file_extensions, crawl=crawl
        )

    elif isinstance(bigwig_path, Iterable):
        return [
            path
            for element in bigwig_path
            for path in interpret_path(
                element, file_extensions=file_extensions, crawl=crawl
            )
        ]
    raise ValueError(
        f"Can not interpret {bigwig_path} as path or a collection of paths."
    )


class PyBigWigCollection:
    def __init__(self, bigwig_path):
        self.bigwig_paths = sorted(interpret_path(bigwig_path))
        self.bigwigs = None

    def init_bigwig_filehandles(self):
        self.bigwigs = [pyBigWig.open(str(bw)) for bw in self.bigwig_paths]

    def get_batch(self, chromosomes, starts, ends):
        if self.bigwigs is None:
            self.init_bigwig_filehandles()
        out = np.stack(
            [
                bw.values(chrom, start, end, numpy=True)
                for chrom, start, end in zip(chromosomes, starts, ends)
                for bw in self.bigwigs
            ]
        )

        out = out.reshape(
            (out.shape[0] // len(self.bigwigs), len(self.bigwigs), out.shape[1])
        )

        return out


class Dataset(IterableDataset):

    """Just a wrapper."""

    def __init__(self, collection, chromosomes, starts, ends):
        self.collection = collection
        self.chromosomes = chromosomes
        self.starts = starts
        self.ends = ends

    def init_collection(self, _):
        self.collection.init_bigwig_filehandles()

    def __len__(self):
        return 128

    def __iter__(self):
        return self

    def __next__(self):
        return self.collection.get_batch(self.chromosomes, self.starts, self.ends)


def some_intervals():
    df = pd.read_csv("example_data/benchmark_positions.tsv", sep="\t")
    start = df["center"].values - 500
    return list(df["chr"]), start, start + 1000


def run_benchmark(batch_sizes, chromosomes, starts, ends, workers):
    for n_workers in workers:
        for batch_size in batch_sizes:
            # print("batch_size", batch_size)

            chrom = chromosomes[:batch_size]
            start = starts[:batch_size]
            end = ends[:batch_size]

            collection = PyBigWigCollection(config.bigwig_dir)

            dataset = Dataset(collection, chrom, start, end)

            # batch_size is set to None because the Dataset returns
            # whole batches at once.
            dataloader = DataLoader(
                dataset,
                num_workers=n_workers,
                batch_size=None,
                worker_init_fn=dataset.init_collection,
            )

            elapsed = []
            start_time = time.perf_counter()
            for i, _ in enumerate(dataloader):
                end_time = time.perf_counter()
                elapsed.append(end_time - start_time)
                start_time = end_time
                if i > 512:
                    break

            print(n_workers, batch_size, np.mean(elapsed), np.std(elapsed))


def run_all_benchmarks(bigwig_path=config.bigwig_dir):
    chromosomes, starts, ends = some_intervals()
    batch_sizes = [32]

    print("Using pybigwig and PyTorch dataloader with multiple number of workers:")
    run_benchmark(
        batch_sizes, chromosomes, starts, ends, workers=[0, 1, 2, 4, 8, 16, 32]
    )


if __name__ == "__main__":
    run_all_benchmarks()
