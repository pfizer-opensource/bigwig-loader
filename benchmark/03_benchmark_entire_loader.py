"""
This runs the entire loader producing the input and target
values a neural network can be trained on. In this file
pyBigWig is used as the library processing the bigwig
data.

A more plain benchmark can be found in benchmark_minimal.py
In that file, the same set of intervals is queried over and
over. This eliminates some of the influence of IO on the
numbers and also does not include things that have nothing
to do with bigwig, like the sampling and looking up the
sequence.
"""

import time

import numpy as np
import pandas as pd

from bigwig_loader import config
from bigwig_loader.dataset import BigWigDataset


def run(batch_size):
    train_regions = pd.read_csv("train_regions.tsv", sep="\t")
    print("Loading from:", config.bigwig_dir)

    dataset = BigWigDataset(
        regions_of_interest=train_regions,
        collection=config.bigwig_dir,
        reference_genome_path=config.reference_genome,
        sequence_length=1000,
        center_bin_to_predict=1000,
        batch_size=batch_size,
        batches_per_epoch=10,
        maximum_unknown_bases_fraction=0.1,
        sequence_encoder="onehot",
    )

    elapsed = []
    for i, (sequence, target) in enumerate(dataset):
        if i == 0:
            start_time = time.perf_counter()
            continue
        end_time = time.perf_counter()
        elapsed.append(end_time - start_time)
        start_time = end_time
    print(batch_size, np.mean(elapsed), np.std(elapsed))


if __name__ == "__main__":
    run(batch_size=2048)
