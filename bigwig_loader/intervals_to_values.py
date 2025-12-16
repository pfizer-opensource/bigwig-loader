import logging
import math
from datetime import datetime
from math import isnan
from pathlib import Path
from typing import Literal

import cupy as cp

from bigwig_loader.default_value import replace_out_tensor_if_needed
from bigwig_loader.searchsorted import interval_searchsorted

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CUDA_KERNEL_DIR = Path(__file__).parent.parent / "cuda_kernels"

DEBUG_DUMP_DIR = Path("./bigwig_loader_debug")


def save_debug_tensors(
    prefix: str,
    array_start: cp.ndarray,
    array_end: cp.ndarray,
    array_value: cp.ndarray,
    query_starts: cp.ndarray,
    query_ends: cp.ndarray,
    found_starts: cp.ndarray,
    found_ends: cp.ndarray,
    out: cp.ndarray,
    scaling_factors: cp.ndarray | None,
    metadata: dict[str, str | int],
) -> Path:
    """Save tensors to disk for debugging."""
    import numpy as np

    DEBUG_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dump_dir = DEBUG_DUMP_DIR / f"{prefix}_{timestamp}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays as numpy files (move to CPU first)
    np.save(dump_dir / "array_start.npy", array_start.get())
    np.save(dump_dir / "array_end.npy", array_end.get())
    np.save(dump_dir / "array_value.npy", array_value.get())
    np.save(dump_dir / "query_starts.npy", query_starts.get())
    np.save(dump_dir / "query_ends.npy", query_ends.get())
    np.save(dump_dir / "found_starts.npy", found_starts.get())
    np.save(dump_dir / "found_ends.npy", found_ends.get())
    np.save(dump_dir / "out.npy", out.get())
    if scaling_factors is not None:
        np.save(dump_dir / "scaling_factors.npy", scaling_factors.get())

    # Save metadata
    import json

    with open(dump_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved debug tensors to {dump_dir}")
    return dump_dir


class Kernel:
    def __init__(
        self, cudafile: Path = CUDA_KERNEL_DIR / "intervals_to_values_contiguous.cu"
    ) -> None:
        with open(cudafile) as f:
            self.kernel_code = f.read()

        self.kernel_names = {
            "float32": "intervals_to_values_float32",
            "bfloat16": "intervals_to_values_bfloat16",
        }
        self._kernels: dict[str, cp.RawKernel] = {}

    def get_kernel_by_dtype(
        self, dtype: Literal["float32", "bfloat16"]
    ) -> cp.RawKernel:
        if dtype in self._kernels:
            return self._kernels[dtype]
        kernel = cp.RawKernel(self.kernel_code, self.kernel_names[dtype])
        kernel.compile()
        self._kernels[dtype] = kernel
        return kernel

    def __call__(  # type: ignore[no-untyped-def]
        self, *args, dtype: Literal["float32", "bfloat16"], **kwargs
    ) -> cp.ndarray:
        kernel = self.get_kernel_by_dtype(dtype=dtype)
        return kernel(*args, **kwargs)


cuda_kernel = Kernel()


def intervals_to_values(
    array_start: cp.ndarray,
    array_end: cp.ndarray,
    array_value: cp.ndarray,
    query_starts: cp.ndarray,
    query_ends: cp.ndarray,
    found_starts: cp.ndarray | None = None,
    found_ends: cp.ndarray | None = None,
    sizes: cp.ndarray | None = None,
    window_size: int = 1,
    default_value: float = 0.0,
    out: cp.ndarray | None = None,
    scaling_factors: cp.ndarray | None = None,
    dtype: Literal["float32", "bfloat16"] = "float32",
) -> cp.ndarray:
    """
    This function converts intervals to values. It can do this for multiple tracks at once.
    When multiple tracks are given, track_starts, track_ends and track_values are expected
    to be concatenated arrays of the individual tracks. The sizes array is used to indicate
    where the individual tracks start and end.

    When none of found_starts, found_ends or sizes are given, it is assumed that there is only
    one track.

    When the sequence length is not a multiple of window_size, the output length will
    be sequence_length // window_size, ignoring the last "incomplete" window.

    Output shape: batch_size x sequence_length x n_tracks (no transpose needed!)

    Args:
        array_start: array of length sum(sizes) with the start positions of the intervals
        array_end: array of length sum(sizes) with the end positions of the intervals
        array_value: array of length sum(sizes) with the value for those intervals
        query_starts: array of length batch_size with the (genomic) start positions of each batch element
        query_ends: array of length batch_size with the (genomic) end positions of each batch element
        found_starts: result of searchsorted (if precalculated). Indices into track_starts.
        found_ends: result of searchsorted (if precalculated). Indices into track_ends.
        sizes: number of elements in track_starts/track_ends/track_values for each track.
            Only needed when found_starts and found_ends are not given.
        window_size: size in basepairs to average over (default: 1)
        default_value: value to use for regions where no data is specified (default: 0.0)
        out: array of size batch_size x sequence_length x n_tracks to store the output.
        scaling_factors: optional array of shape (n_tracks,) to scale each track's values
        dtype: output dtype, either 'float32' or 'bfloat16'
    Returns:
        out: array of size batch_size x sequence_length x n_tracks

    """
    if cp.unique(query_ends - query_starts).size != 1:
        raise ValueError(
            "All queried intervals should have the same length. Found lengths: ",
            cp.unique(query_ends - query_starts),
        )
    sequence_length = (query_ends[0] - query_starts[0]).item()

    if (found_starts is None or found_ends is None) and sizes is None:
        sizes = cp.asarray([len(array_start)], dtype=array_start.dtype)

    if found_starts is None or found_ends is None:
        found_starts, found_ends = interval_searchsorted(
            array_start,
            array_end,
            query_starts,
            query_ends,
            sizes,
            absolute_indices=True,
        )

    reduced_sequence_length = sequence_length // window_size
    batch_size = len(query_starts)
    num_tracks = found_starts.shape[0]

    logger.debug("--------------------------")
    if out is not None:
        logger.debug(f"Out tensor before replace: {out.shape}, dtype: {out.dtype}")

    out = replace_out_tensor_if_needed(
        out,
        batch_size=batch_size,
        sequence_length=reduced_sequence_length,
        number_of_tracks=num_tracks,
        default_value=default_value,
        dtype=dtype,
        reset_values=True,
    )

    logger.debug(f"Out tensor after replace: {out.shape}, dtype: {out.dtype}")

    max_number_intervals = min(
        sequence_length, (found_ends - found_starts).max().item()
    )

    if window_size == 1:
        n_threads_needed = batch_size * max_number_intervals * num_tracks
        grid_size, block_size = get_grid_and_block_size(n_threads_needed)
    else:
        n_threads_needed = batch_size * num_tracks
        grid_size, block_size = get_grid_and_block_size(n_threads_needed)

    logging.debug(
        f"batch_size: {batch_size}\nmax_number_intervals: {max_number_intervals}\ngrid_size: {grid_size}\nblock_size: {block_size}"
    )

    query_starts = cp.ascontiguousarray(query_starts)
    query_ends = cp.ascontiguousarray(query_ends)
    found_starts = cp.ascontiguousarray(found_starts)
    found_ends = cp.ascontiguousarray(found_ends)
    array_start = cp.ascontiguousarray(array_start)
    array_end = cp.ascontiguousarray(array_end)
    array_value = cp.ascontiguousarray(array_value)
    default_value_isnan = isnan(default_value)

    # Handle scaling factors - convert to float32 if provided
    if scaling_factors is not None:
        scaling_factors_arr = cp.ascontiguousarray(scaling_factors, dtype=cp.float32)
        use_scaling_factors = True
    else:
        scaling_factors_arr = cp.empty(
            0, dtype=cp.float32
        )  # Empty array, won't be accessed
        use_scaling_factors = False

    logger.debug("intervals_to_values inputs:")
    logger.debug(
        f"  array_start shape: {array_start.shape}, dtype: {array_start.dtype}"
    )
    logger.debug(f"  array_end shape: {array_end.shape}, dtype: {array_end.dtype}")
    logger.debug(
        f"  array_value shape: {array_value.shape}, dtype: {array_value.dtype}"
    )
    logger.debug(
        f"  found_starts shape: {found_starts.shape}, min: {int(found_starts.min())}, max: {int(found_starts.max())}"
    )
    logger.debug(
        f"  found_ends shape: {found_ends.shape}, min: {int(found_ends.min())}, max: {int(found_ends.max())}"
    )
    logger.debug(f"  out shape: {out.shape}")
    logger.debug(
        f"  array_start min/max: {int(array_start.min())}, {int(array_start.max())}"
    )
    logger.debug(f"  array_end min/max: {int(array_end.min())}, {int(array_end.max())}")
    logger.debug(f"  max_number_intervals: {max_number_intervals}")
    logger.debug(
        f"  num_tracks: {num_tracks}, batch_size: {batch_size}, sequence_length: {sequence_length}"
    )

    # ========== DEBUG: Additional validation ==========
    # Check for cases where found_starts > found_ends (invalid ranges)
    invalid_mask = found_starts > found_ends
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        logger.warning(f"  {n_invalid} tracks have found_starts > found_ends")
        invalid_indices = cp.where(invalid_mask.any(axis=1))[0]
        logger.warning(
            f"  Invalid track indices (first 10): {invalid_indices[:10].get()}"
        )
        for idx in invalid_indices[:5].get():
            logger.warning(
                f"    track {idx}: found_start={int(found_starts[idx, 0])}, found_end={int(found_ends[idx, 0])}"
            )

    # Check query range vs data range
    logger.debug(f"  query_starts: {query_starts.get()}")
    logger.debug(f"  query_ends: {query_ends.get()}")

    # Check if any intervals are negative length
    interval_lengths = found_ends.astype(cp.int64) - found_starts.astype(cp.int64)
    if int(interval_lengths.min()) < 0:
        logger.warning(
            f"  Negative interval lengths detected! min: {int(interval_lengths.min())}"
        )

    # Check grid/block calculation
    logger.debug(f"  grid_size: {grid_size}, block_size: {block_size}")
    logger.debug(f"  n_threads_needed: {n_threads_needed}")

    try:
        cuda_kernel(
            (grid_size,),
            (block_size,),
            (
                query_starts,
                query_ends,
                found_starts,
                found_ends,
                array_start,
                array_end,
                array_value,
                num_tracks,
                batch_size,
                sequence_length,
                max_number_intervals,
                window_size,
                cp.float32(default_value),
                default_value_isnan,
                scaling_factors_arr,
                use_scaling_factors,
                out,
            ),
            dtype=dtype,
        )
    except Exception as e:
        logger.error(f"CUDA kernel failed: {e}")
        try:
            # Try to save tensors - might fail if CUDA context is corrupted
            save_debug_tensors(
                "kernel_error",
                array_start,
                array_end,
                array_value,
                query_starts,
                query_ends,
                found_starts,
                found_ends,
                out,
                scaling_factors_arr if use_scaling_factors else None,
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "num_tracks": num_tracks,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "window_size": window_size,
                    "max_number_intervals": max_number_intervals,
                    "grid_size": grid_size,
                    "block_size": block_size,
                },
            )
        except Exception as save_error:
            logger.error(f"Failed to save debug tensors: {save_error}")
        raise
    return out


def get_grid_and_block_size(n_threads: int) -> tuple[int, int]:
    n_blocks_needed = math.ceil(n_threads / 512)
    if n_blocks_needed == 1:
        threads_per_block = n_threads
    else:
        threads_per_block = 512
    return n_blocks_needed, threads_per_block


def kernel_in_python_with_window(
    grid_size: tuple[int],
    block_size: tuple[int],
    args: tuple[
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        int,
        int,
        int,
        int,
        float,
        bool,
        cp.ndarray,
    ],
) -> cp.ndarray:
    """Equivalent in python to cuda_kernel_with_window. Just for debugging."""

    (
        query_starts,
        query_ends,
        found_starts,
        found_ends,
        track_starts,
        track_ends,
        track_values,
        num_tracks,
        batch_size,
        sequence_length,
        max_number_intervals,
        window_size,
        default_value,
        default_value_isnan,
        out,
    ) = args

    _grid_size = grid_size[0]
    _block_size = block_size[0]

    query_starts = query_starts.get().tolist()
    query_ends = query_ends.get().tolist()

    # flattening this because that's how we get it in cuda
    found_starts = found_starts.flatten().get().tolist()
    found_ends = found_ends.flatten().get().tolist()

    track_starts = track_starts.get().tolist()
    track_ends = track_ends.get().tolist()
    track_values = track_values.get().tolist()

    n_threads = _grid_size * _block_size

    print(n_threads)

    # this should be integer
    reduced_dim = sequence_length // window_size
    print("sequence_length")
    print(sequence_length)
    print("reduced_dim")
    print(reduced_dim)

    out_vector = [default_value] * reduced_dim * batch_size * num_tracks

    for thread in range(n_threads):
        batch_index = thread % batch_size
        track_index = (thread // batch_size) % num_tracks
        i = thread % (batch_size * num_tracks)

        print("\n\n\n######")
        print(f"NEW thread {thread}")
        print("batch_index", batch_index)
        print("track_index", track_index)
        print("i", i)

        # if i < batch_size * num_tracks:
        found_start_index = found_starts[i]
        found_end_index = found_ends[i]
        query_start = query_starts[batch_index]
        query_end = query_ends[batch_index]

        cursor = found_start_index
        window_index = 0
        summation = 0.0
        valid_count = 0

        # cursor moves through the rows of the bigwig file
        # window_index moves through the sequence

        while cursor < found_end_index and window_index < reduced_dim:
            print("-----")
            print("cursor:", cursor)
            window_start = window_index * window_size
            window_end = window_start + window_size
            print(f"working on values in output window {window_start} - {window_end}")
            print(
                f"Corresponding to the genomic loc   {query_start + window_start} - {query_start + window_end}"
            )

            interval_start = track_starts[cursor]
            interval_end = track_ends[cursor]

            print("bigwig interval_start", "bigwig interval_end", "bigwig value")
            print(interval_start, interval_end, track_values[cursor])

            start_index = max(interval_start - query_start, 0)
            end_index = min(interval_end, query_end) - query_start
            print("start index", start_index)

            if start_index >= window_end:
                if default_value_isnan:
                    if valid_count > 0:
                        out_vector[i * reduced_dim + window_index] = (
                            summation / valid_count
                        )
                    else:
                        out_vector[i * reduced_dim + window_index] = default_value
                else:
                    summation = summation + (window_size - valid_count) * default_value
                    out_vector[i * reduced_dim + window_index] = summation / window_size
                summation = 0.0
                valid_count = 0
                window_index += 1
                print("CONTINUE")
                continue

            number = min(window_end, end_index) - max(window_start, start_index)

            if number > 0:
                print(
                    f"Add {number} x {track_values[cursor]} = {number * track_values[cursor]} to summation"
                )
                summation += number * track_values[cursor]
                print(f"Summation = {summation}")
                valid_count += number

            print("end_index", "window_end")
            print(end_index, window_end)

            # calculate average, reset summation and move to next window
            if end_index >= window_end or cursor + 1 >= found_end_index:
                if end_index >= window_end:
                    print(
                        "end_index >= window_end \t\t calculate average, reset summation and move to next window"
                    )
                else:
                    print(
                        "cursor + 1 >= found_end_index \t\t calculate average, reset summation and move to next window"
                    )
                # out_vector[i * reduced_dim + window_index] = summation / window_size
                if default_value_isnan:
                    if valid_count > 0:
                        out_vector[i * reduced_dim + window_index] = (
                            summation / valid_count
                        )
                    else:
                        out_vector[i * reduced_dim + window_index] = default_value
                else:
                    summation = summation + (window_size - valid_count) * default_value
                    out_vector[i * reduced_dim + window_index] = summation / window_size
                summation = 0.0
                valid_count = 0
                window_index += 1
            # move cursor
            if end_index < window_end:
                print("move cursor")
                cursor += 1
            print("current out state:", out_vector)
            print(
                cp.reshape(
                    cp.asarray(out_vector), (num_tracks, batch_size, reduced_dim)
                )
            )

    return cp.reshape(cp.asarray(out_vector), (num_tracks, batch_size, reduced_dim))


def kernel_in_python(
    grid_size: int,
    block_size: int,
    args: tuple[
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        int,
        int,
        int,
        cp.ndarray,
        int,
    ],
) -> cp.ndarray:
    """Equivalent in python to cuda_kernel. Just for debugging."""

    (
        query_starts,
        query_ends,
        found_starts,
        found_ends,
        track_starts,
        track_ends,
        track_values,
        batch_size,
        sequence_length,
        max_number_intervals,
        _,
        window_size,
    ) = args

    query_starts = query_starts.get().tolist()
    query_ends = query_ends.get().tolist()

    found_starts = found_starts.get().tolist()
    found_ends = found_ends.get().tolist()
    track_starts = track_starts.get().tolist()
    track_ends = track_ends.get().tolist()
    track_values = track_values.get().tolist()

    n_threads = grid_size * block_size

    out = [0.0] * sequence_length * batch_size

    for thread in range(n_threads):
        i = thread % batch_size
        j = (thread // batch_size) % max_number_intervals
        # k = thread // (batch_size * max_number_intervals)
        # print("---")
        # print(i, j)

        if i < batch_size:
            found_start_index = found_starts[i]
            found_end_index = found_ends[i]
            query_start = query_starts[i]
            query_end = query_ends[i]

            cursor = found_start_index + j
            # print("cursor", cursor)

            if cursor < found_end_index:
                interval_start = track_starts[cursor]
                interval_end = track_ends[cursor]
                start_index = max(interval_start - query_start, 0)
                end_index = (
                    (i * sequence_length) + min(interval_end, query_end) - query_start
                )
                start_position = (i * sequence_length) + start_index
                for position in range(start_position, end_index):
                    print("position", position, track_values[cursor])
                    out[position] = track_values[cursor]
        # print(out)

    # print(out)
    out = cp.reshape(cp.asarray(out), (batch_size, sequence_length))
    return out
