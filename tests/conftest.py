import logging
import random

import cupy as cp
import pandas as pd
import pytest
import torch

from bigwig_loader import config
from bigwig_loader.download_example_data import get_example_bigwigs_files
from bigwig_loader.download_example_data import get_reference_genome

try:
    from bigwig_loader.collection import BigWigCollection
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


def all_close(expected, values, dtype):
    """Compare arrays/tensors with dtype-appropriate tolerances.

    For bfloat16, uses looser tolerances due to lower precision.
    For float32, uses standard tolerances.
    """
    if dtype == "float32":
        return cp.allclose(expected, values, equal_nan=True)
    elif dtype == "bfloat16":
        expected = convert_to_torch_bfloat16(expected)
        values = convert_to_torch_bfloat16(values)
        return torch.allclose(expected, values, rtol=1e-2, atol=1e-2, equal_nan=True)
    raise ValueError(f"Unsupported data type: {dtype}")


def convert_to_torch_bfloat16(tensor):
    """Convert a CuPy array to PyTorch bfloat16 tensor.

    For uint16 tensors (bfloat16 bits), reinterprets bits as bfloat16.
    For float32 tensors, converts to bfloat16.
    """

    torch_tensor = torch.as_tensor(tensor)

    if tensor.dtype == cp.uint16:
        # Reinterpret uint16 bits as bfloat16 using view
        # This treats the uint16 bit pattern as if it were bfloat16
        torch_tensor = torch_tensor.view(torch.bfloat16)
    elif tensor.dtype == cp.float32:
        # Convert float32 to bfloat16
        torch_tensor = torch_tensor.to(torch.bfloat16)
    else:
        # Convert other types to float32 first, then to bfloat16
        torch_tensor = torch_tensor.to(torch.float32).to(torch.bfloat16)
    return torch_tensor


def uint16_to_float32(tensor):
    """Convert uint16 array (containing bfloat16 bits) to float32 values.

    This interprets the uint16 bit pattern as bfloat16, then converts to float32.
    Keeps data on GPU throughout the operation.
    """
    torch_tensor = convert_to_torch_bfloat16(tensor)
    # Convert to float32 to get actual values (stays on GPU)
    float32_tensor = torch_tensor.to(torch.float32)
    # Convert back to CuPy (stays on GPU)
    return cp.asarray(float32_tensor)
