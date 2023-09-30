import logging

import pytest

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.download_example_data import get_example_bigwigs_files
from bigwig_loader.download_example_data import get_reference_genome

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
