import logging
import random

import pandas as pd
import pytest

from bigwig_loader import config

try:
    from bigwig_loader.collection import BigWigCollection
    from bigwig_loader.download_example_data import get_example_bigwigs_files
    from bigwig_loader.download_example_data import get_reference_genome
except ImportError:
    logging.warning(
        "Can not import from bigwig_loader.collection without cupy installed"
    )

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def example_data_dir():
    return config.example_data_dir


@pytest.fixture
def reference_genome_path():
    return get_reference_genome()


@pytest.fixture(scope="session")
def bigwig_path():
    return get_example_bigwigs_files()


@pytest.fixture
def collection(bigwig_path):
    collection = BigWigCollection(bigwig_path, first_n_files=2)
    return collection


@pytest.fixture
def merged_intervals(collection):
    return collection.intervals(
        ["chr3", "chr4", "chr5"], exclude_chromosomes=["chr4"], merge=True, threshold=2
    )


@pytest.fixture
def example_intervals(collection) -> tuple[list[str], list[int], list[int]]:
    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]
    chrom, start, end = (
        list(df["chr"]),
        list(df["center"] - 500),
        list(df["center"] + 500),
    )

    shuffle_index = list(range(len(start)))
    random.Random(56).shuffle(shuffle_index)
    chrom = [chrom[i] for i in shuffle_index]
    start = [start[i] for i in shuffle_index]
    end = [end[i] for i in shuffle_index]
    return chrom, start, end
