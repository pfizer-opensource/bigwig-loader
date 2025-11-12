from math import isnan
from typing import Literal

import cupy as cp


def get_default_value(value: float, dtype: str) -> cp.uint16 | cp.float32:
    """bfloat16 is not supported by cupy yet,
    so we use uint16. This only works for the
    final output, because we have a custom
    cuda kernel (which does support bfloat16)
    populating the final out tensor.
    """
    if dtype == "bfloat16":
        if value == 0:
            return cp.uint16(0)
        if isnan(value):
            return cp.uint16(0x7FC0)
        else:
            raise ValueError(
                "When using bfloat16, please do not use"
                "any other default value than 0 or NaN."
            )
    if dtype == "float32" or dtype == cp.float32:
        return cp.float32(value)
    raise ValueError("Only bfloat16 or float32 are supported.")


def create_output_tensor(
    batch_size: int,
    sequence_length: int,
    number_of_tracks: int,
    default_value: float = 0.0,
    dtype: Literal["bfloat16", "float32"] = "float32",
) -> cp.ndarray:
    """Get output tensor in the correct shape: batch_size x sequence_length x n_tracks"""
    shape = (batch_size, sequence_length, number_of_tracks)
    if dtype == "bfloat16":
        out_dtype = cp.uint16
    elif dtype == "float32":
        out_dtype = cp.float32
    else:
        raise ValueError("Only bfloat16 or float32 are supported.")
    converted_default_value = get_default_value(value=default_value, dtype=dtype)

    return cp.full(shape, converted_default_value, dtype=out_dtype)


def output_tensor_correct_format(
    tensor: cp.ndarray,
    batch_size: int,
    sequence_length: int,
    number_of_tracks: int,
    dtype: str,
) -> bool:
    if tensor.shape != (batch_size, sequence_length, number_of_tracks):
        return False
    if dtype not in ("bfloat16", "float32"):
        return False
    if dtype == "bfloat16" and tensor.dtype != cp.dtype("uint16"):
        return False
    if (dtype == "float32" or dtype == cp.float32) and tensor.dtype != cp.dtype(
        "float32"
    ):
        return False
    return True


def fill(
    tensor: cp.ndarray,
    default_value: float,
    dtype: Literal["bfloat16", "float32"] = "float32",
) -> cp.ndarray:
    tensor.fill(get_default_value(default_value, dtype=dtype))
    return tensor


def replace_out_tensor_if_needed(
    tensor: cp.ndarray | None,
    batch_size: int,
    sequence_length: int,
    number_of_tracks: int,
    default_value: float = 0.0,
    dtype: Literal["bfloat16", "float32"] = "float32",
    reset_values: bool = False,
) -> cp.ndarray:
    """Replace tensor if needed. First check if the given tensor is correct format.

    If the tensor is None or has incorrect shape/dtype, create a new one.
    Otherwise, return the existing tensor.
    """
    if tensor is None:
        return create_output_tensor(
            batch_size=batch_size,
            sequence_length=sequence_length,
            number_of_tracks=number_of_tracks,
            default_value=default_value,
            dtype=dtype,
        )

    if output_tensor_correct_format(
        tensor=tensor,
        batch_size=batch_size,
        sequence_length=sequence_length,
        number_of_tracks=number_of_tracks,
        dtype=dtype,
    ):
        if reset_values:
            tensor = fill(tensor, default_value=default_value, dtype=dtype)
        return tensor

    return create_output_tensor(
        batch_size=batch_size,
        sequence_length=sequence_length,
        number_of_tracks=number_of_tracks,
        default_value=default_value,
        dtype=dtype,
    )
