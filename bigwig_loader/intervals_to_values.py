import logging
import math
from pathlib import Path

import cupy as cp

from bigwig_loader.searchsorted import interval_searchsorted

CUDA_KERNEL_DIR = Path(__file__).parent.parent / "cuda_kernels"

_zero = cp.asarray(0.0, dtype=cp.float32).item()


def get_cuda_kernel() -> str:
    with open(CUDA_KERNEL_DIR / "intervals_to_values.cu") as f:
        kernel_code = f.read()
    return kernel_code


cuda_kernel = cp.RawKernel(get_cuda_kernel(), "intervals_to_values")
cuda_kernel.compile()


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
    out: cp.ndarray | None = None,
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


    Args:
        array_start: array of length sum(sizes) with the start positions of the intervals
        array_end: array of length sum(sizes) with the end positions of the intervals
        array_value: array of length sum(sizes) with the value for those intervals
        query_starts: array of length batch_size with the (genomic) start positions of each batch element
        query_ends: array of length batch_size with the (genomic) end positions of each batch element
        out: array of size n_tracks x batch_size x sequence_length to store the output
        found_starts: result of searchsorted (if precalculated). Indices into track_starts.
        found_ends: result of searchsorted (if precalculated). Indices into track_ends.
        sizes: number of elements in track_starts/track_ends/track_values for each track.
            Only needed when found_starts and found_ends are not given.
        window_size: size in basepairs to average over (default: 1)
    Returns:
        out: array of size n_tracks x batch_size x sequence_length

    """
    if cp.unique(query_ends - query_starts).size != 1:
        raise ValueError(
            "All queried intervals should have the same length. Found lengths: ",
            cp.unique(query_ends - query_starts),
        )
    sequence_length = (query_ends[0] - query_starts[0]).item()

    if (found_starts is None or found_ends is None) and sizes is None:
        # just one size, which is the length of the entire track_starts/tracks_ends/tracks_values
        sizes = cp.asarray([len(array_start)], dtype=array_start.dtype)

    if found_starts is None or found_ends is None:
        # n_subarrays x n_queries
        found_starts, found_ends = interval_searchsorted(
            array_start,
            array_end,
            query_starts,
            query_ends,
            sizes,
            absolute_indices=True,
        )

    if out is None:
        out = cp.zeros(
            (found_starts.shape[0], len(query_starts), sequence_length // window_size),
            dtype=cp.float32,
        )
    else:
        out *= _zero

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

    query_starts = cp.ascontiguousarray(query_starts)
    query_ends = cp.ascontiguousarray(query_ends)
    found_starts = cp.ascontiguousarray(found_starts)
    found_ends = cp.ascontiguousarray(found_ends)
    array_start = cp.ascontiguousarray(array_start)
    array_end = cp.ascontiguousarray(array_end)
    array_value = cp.ascontiguousarray(array_value)

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
        num_tracks,
        batch_size,
        sequence_length,
        max_number_intervals,
        window_size,
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

    out_vector = [0.0] * reduced_dim * batch_size * num_tracks

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
        summation = 0

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
                print("CONTINUE")
                out_vector[i * reduced_dim + window_index] = summation / window_size
                summation = 0
                window_index += 1
                continue

            number = min(window_end, end_index) - max(window_start, start_index)

            print(
                f"Add {number} x {track_values[cursor]} = {number * track_values[cursor]} to summation"
            )
            summation += number * track_values[cursor]
            print(f"Summation = {summation}")

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
                out_vector[i * reduced_dim + window_index] = summation / window_size
                summation = 0
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
