from typing import Literal

import cupy as cp
from cupy import _core
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


_searchsorted_kernel = _core.ElementwiseKernel(
    "S x, S index, raw int64 starts, raw T sizes, raw T all_bins, bool side_is_right, "
    "bool assume_increasing",
    "int64 y",
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
    sizes: cp.ndarray,
    side: Literal["left", "right"] = "left",
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
    Returns:
        And array of size n_subarrays x n_queries with insertion indices.

    """

    start_indices = cp.pad(cp.cumsum(sizes, dtype=cp.int64), (1, 0))[:-1]
    n_subarrays = len(sizes)
    n_queries = len(queries)
    idx = cp.arange(n_subarrays, dtype=queries.dtype)[:, cp.newaxis]
    queries = queries[cp.newaxis, :]
    output = cp.zeros((n_subarrays, n_queries), dtype=cp.int64)

    return _searchsorted_kernel(
        queries, idx, start_indices, sizes, array, side == "right", True, output
    )
