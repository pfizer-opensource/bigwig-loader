"""
"""

import time

import numpy as np
import pandas as pd

from bigwig_loader import config
from bigwig_loader import intervals_to_values_gpu
from bigwig_loader.collection import BigWigCollection


def some_intervals():
    df = pd.read_csv("example_data/benchmark_positions.tsv", sep="\t")
    start = df["center"].values - 5000
    return list(df["chr"]), start, start + 10000


def run_benchmark(collection, batch_sizes, chromosomes, starts, ends):
    for batch_size in batch_sizes:
        # print("batch_size", batch_size)

        chrom = chromosomes[:batch_size]
        start = starts[:batch_size]
        end = ends[:batch_size]

        # burn in
        for _ in range(20):
            collection.get_batch(chrom, start, end)

        elapsed = []
        start_time = time.perf_counter()
        for _ in range(20):
            collection.get_batch(chrom, start, end)
            end_time = time.perf_counter()
            elapsed.append(end_time - start_time)
            start_time = end_time
        # print("Seconds per batch:", np.mean(elapsed))
        # print("standard deviation:", np.std(elapsed))
        print(batch_size, np.mean(elapsed), np.std(elapsed))


def run_all_benchmarks(bigwig_path=config.bigwig_dir):
    print("Loading from:", config.bigwig_dir)
    bigwig_loader_collection = BigWigCollection(bigwig_path, first_n_files=None)
    chromosomes, starts, ends = some_intervals()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64] + [128 * i for i in range(1, 48)]

    print("routing to different cupy kernels:")
    run_benchmark(bigwig_loader_collection, batch_sizes, chromosomes, starts, ends)

    intervals_to_values_gpu.ROUTE_KERNELS = False
    print("routing to same cupy kernel:")
    run_benchmark(bigwig_loader_collection, batch_sizes, chromosomes, starts, ends)


if __name__ == "__main__":
    run_all_benchmarks()
