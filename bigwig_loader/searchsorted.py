from typing import Literal

import cupy as cp
from cupy._sorting.search import _searchsorted_code

_preamble = """
template<typename T>
__device__ bool _isnan(T val) {
    return val != val;
}
"""

_hip_preamble = r"""
#ifdef __HIP_DEVICE_COMPILE__
  #define no_thread_divergence(do_work, to_return) \
    if (!is_done) {                                \
      do_work;                                     \
      is_done = true;                              \
    }
#else
  #define no_thread_divergence(do_work, to_return) \
    do_work;                                       \
    if (to_return) { return; }
#endif
"""


_searchsorted_kernel = cp.ElementwiseKernel(
    "S x, S index, raw uint32 starts, raw T sizes, raw T all_bins, bool side_is_right, "
    "bool assume_increasing",
    "uint32 y",
    """
    int start = starts[index];
    int n_bins = sizes[index];
    const T* bins = &all_bins[start];

    """
    + _searchsorted_code,
    name="cupy_searchsorted_kernel",
    preamble=_preamble + _hip_preamble,
)


def searchsorted(
    array: cp.ndarray,
    queries: cp.ndarray,
    sizes: cp.ndarray | None = None,
    start_indices: cp.ndarray | None = None,
    side: Literal["left", "right"] = "left",
    absolute_indices: bool = True,
) -> cp.ndarray:
    """
    This is a version of search sorted does the searchsorted operation on
    multiple subarrays at once (for the same queries to find the insertion
    points for). Where each subarray starts is indicated by start_indices.

    Args:
        array: 1D Input array. Is expected to consist of subarrays that are
            sorted in ascending order.
        queries: Values to find the insertion indices for in subarrays of array.
        start_indices: Indices of the starts of the subarrays in array.
        sizes: Sizes of the subarrays.
        side: If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.
        absolute_indices: whether to give the indices with respect to the entire
            array (True) or for the subarrays (False).
    Returns:
        And array of size n_subarrays x n_queries with insertion indices.

    """
    start_indices, sizes = starts_and_sizes(start_indices, sizes)

    n_subarrays = len(sizes)
    n_queries = len(queries)
    idx = cp.arange(n_subarrays, dtype=queries.dtype)[:, cp.newaxis]
    queries = queries[cp.newaxis, :]
    output = cp.zeros((n_subarrays, n_queries), dtype=cp.uint32)

    result = _searchsorted_kernel(
        queries, idx, start_indices, sizes, array, side == "right", True, output
    )
    if absolute_indices:
        return result + start_indices[:, cp.newaxis]
    return result


def interval_searchsorted(
    array_start: cp.ndarray,
    array_end: cp.ndarray,
    query_starts: cp.ndarray,
    query_ends: cp.ndarray,
    sizes: cp.ndarray | None = None,
    start_indices: cp.ndarray | None = None,
    absolute_indices: bool = True,
) -> tuple[cp.ndarray, cp.ndarray]:
    """This is a convenience function that does searchsorted on both
    the start and end arrays and returns the results.
    """

    start_indices, sizes = starts_and_sizes(start_indices, sizes)

    # n_tracks x n_queries
    found_starts = searchsorted(
        array_end,
        queries=query_starts,
        sizes=sizes,
        side="right",
        absolute_indices=absolute_indices,
    )
    found_ends = searchsorted(
        array_start,
        queries=query_ends,
        sizes=sizes,
        side="left",
        absolute_indices=absolute_indices,
    )

    return found_starts, found_ends


def starts_and_sizes(
    starts: cp.ndarray | None, sizes: cp.ndarray | None
) -> tuple[cp.ndarray, cp.ndarray]:
    if starts is None and sizes is None:
        raise ValueError("Either starts or sizes must be provided.")
    elif starts is None:
        starts = sizes_to_starts(sizes)
    elif sizes is None:
        sizes = starts_to_sizes(starts)
    return starts, sizes


def sizes_to_starts(sizes: cp.ndarray) -> cp.ndarray:
    return cp.pad(cp.cumsum(sizes, dtype=cp.uint32), (1, 0))[:-1]


def starts_to_sizes(starts: cp.ndarray) -> cp.ndarray:
    return (starts[1:] - starts[:-1]).astype(cp.uint32)
