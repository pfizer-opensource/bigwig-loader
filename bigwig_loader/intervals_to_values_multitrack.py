import logging
import math
from pathlib import Path

import cupy as cp

from bigwig_loader.searchsorted import searchsorted

CUDA_KERNEL_DIR = Path(__file__).parent.parent / "cuda_kernels"

_zero = cp.asarray(0.0, dtype=cp.float32).item()


def get_cuda_kernel() -> str:
    with open(CUDA_KERNEL_DIR / "intervals_to_values.cu") as f:
        kernel_code = f.read()
    return kernel_code


cuda_kernel = cp.RawKernel(get_cuda_kernel(), "intervals_to_values")
cuda_kernel.compile()


def intervals_to_values(
    track_starts: cp.ndarray,
    track_ends: cp.ndarray,
    track_values: cp.ndarray,
    query_starts: cp.ndarray,
    query_ends: cp.ndarray,
    out: cp.ndarray,
    found_starts: cp.ndarray | None = None,
    found_ends: cp.ndarray | None = None,
    sizes: cp.ndarray | None = None,
    window_size: int = 1,
) -> cp.ndarray:
    if (found_starts is None or found_ends is None) and sizes is None:
        raise ValueError("Either found_starts and found_ends or sizes must be provided")

    if found_starts is None or found_ends is None:
        found_starts = searchsorted(
            track_ends,
            queries=query_starts,
            sizes=sizes,
            side="right",
            absolute_indices=True,
        )
        found_ends = searchsorted(
            track_starts,
            queries=query_ends,
            sizes=sizes,
            side="left",
            absolute_indices=True,
        )

    out *= _zero

    sequence_length = (query_ends[0] - query_starts[0]).item()

    max_number_intervals = min(
        sequence_length, (found_ends - found_starts).max().item()
    )
    batch_size = query_starts.shape[0]
    num_tracks = found_starts.shape[0]

    if window_size == 1:
        n_threads_needed = batch_size * max_number_intervals * num_tracks
        grid_size, block_size = get_grid_and_block_size(n_threads_needed)
    else:
        n_threads_needed = batch_size * num_tracks
        grid_size, block_size = get_grid_and_block_size(n_threads_needed)

    logging.debug(
        f"batch_size: {batch_size}\nmax_number_intervals: {max_number_intervals}\ngrid_size: {grid_size}\nblock_size: {block_size}"
    )

    cuda_kernel(
        (grid_size,),
        (block_size,),
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
            out,
        ),
    )

    return out


def get_grid_and_block_size(n_threads: int) -> tuple[int, int]:
    n_blocks_needed = math.ceil(n_threads / 512)
    if n_blocks_needed == 1:
        threads_per_block = n_threads
    else:
        threads_per_block = 512
    return n_blocks_needed, threads_per_block


def kernel_in_python_with_window(
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
    """Equivalent in python to cuda_kernel_with_window. Just for debugging."""

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

    # this should be integer
    reduced_dim = sequence_length // window_size

    out = [0.0] * reduced_dim * batch_size

    for thread in range(n_threads):
        i = thread

        if i < batch_size:
            found_start_index = found_starts[i]
            found_end_index = found_ends[i]
            query_start = query_starts[i]
            query_end = query_ends[i]

            cursor = found_start_index
            window_index = 0
            summation = 0

            while cursor < found_end_index and window_index < reduced_dim:
                window_start = window_index * window_size
                window_end = window_start + window_size

                interval_start = track_starts[cursor]
                interval_end = track_ends[cursor]

                start_index = max(interval_start - query_start, 0)
                end_index = min(interval_end, query_end) - query_start

                if start_index >= window_end:
                    window_index += 1
                    continue

                number = min(window_end, end_index) - max(window_start, start_index)

                summation += number * track_values[cursor]
                print("-----")
                print("window_index", "number", "summation")
                print(window_index, number, summation)
                print("interval_start", "interval_end", "value")
                print(interval_start, interval_end, track_values[cursor])

                print("end_index", "window_end")
                print(end_index, window_end)

                # calculate average, reset summation and move to next window
                if end_index >= window_end or cursor + 1 >= found_end_index:
                    print("calculate average, reset summation and move to next window")
                    out[i * reduced_dim + window_index] = summation / window_size
                    summation = 0
                    window_index += 1
                # move cursor
                if end_index < window_end:
                    print("move cursor")
                    cursor += 1

    out = cp.reshape(cp.asarray(out), (batch_size, reduced_dim))
    return out


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
