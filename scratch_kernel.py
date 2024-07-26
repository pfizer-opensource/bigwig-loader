import cupy as cp
from cupy import _core

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


_searchsorted_code = """
    #ifdef __HIP_DEVICE_COMPILE__
    bool is_done = false;
    #endif

    // Array is assumed to be monotonically
    // increasing unless a check is requested with the
    // `assume_increasing = False` parameter.
    // `digitize` allows increasing and decreasing arrays.
    bool inc = true;

    if (!assume_increasing && n_bins >= 2) {
        // In the case all the bins are nan the array is considered
        // to be decreasing in numpy
        inc = (bins[0] <= bins[n_bins-1])
              || (!_isnan<T>(bins[0]) && _isnan<T>(bins[n_bins-1]));
    }

    if (_isnan<S>(x)) {
        long long pos = (inc ? n_bins : 0);
        if (!side_is_right) {
            if (inc) {
                while (pos > 0 && _isnan<T>(bins[pos-1])) {
                    --pos;
                }
            } else {
                while (pos < n_bins && _isnan<T>(bins[pos])) {
                    ++pos;
                }
            }
        }
        no_thread_divergence( y = pos , true )
    }

    bool greater = false;
    if (side_is_right) {
        greater = inc && x >= bins[n_bins-1];
    } else {
        greater = (inc ? x > bins[n_bins-1] : x <= bins[n_bins-1]);
    }
    if (greater) {
        no_thread_divergence( y = n_bins , true )
    }

    long long left = 0;
    // In the case the bins is all NaNs, digitize
    // needs to place all the valid values to the right
    if (!inc) {
        while (_isnan<T>(bins[left]) && left < n_bins) {
            ++left;
        }
        if (left == n_bins) {
            no_thread_divergence( y = n_bins , true )
        }
        if (side_is_right
                && !_isnan<T>(bins[n_bins-1]) && !_isnan<S>(x)
                && bins[n_bins-1] > x) {
            no_thread_divergence( y = n_bins , true )
        }
    }

    long long right = n_bins-1;
    while (left < right) {
        long long m = left + (right - left) / 2;
        bool look_right = true;
        if (side_is_right) {
            look_right = (inc ? bins[m] <= x : bins[m] > x);
        } else {
            look_right = (inc ? bins[m] < x : bins[m] >= x);
        }
        if (look_right) {
            left = m + 1;
        } else {
            right = m;
        }
    }
    no_thread_divergence( y = right , false )
"""


_searchsorted_kernel = _core.ElementwiseKernel(
    "S x, S index, raw T starts, raw T sizes, raw T all_bins, bool side_is_right, "
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


if __name__ == "__main__":
    side = "right"
    background = cp.asarray(
        [
            5,
            10,
            12,
            18,
            1,
            3,
            5,
            7,
            9,
            10,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            4,
            100,
        ],
        dtype=cp.int32,
    )
    queries = cp.asarray(
        [
            [7, 9, 11],
        ],
        dtype=cp.int32,
    )
    idx = cp.asarray([[0], [1], [2], [4]], cp.int32)
    starts = cp.asarray([0, 4, 10, 24], cp.int32)
    sizes = cp.asarray([4, 6, 14, 2], cp.int32)

    output = cp.zeros((4, 3), dtype=cp.int64)

    _searchsorted_kernel(
        queries, idx, starts, sizes, background, side == "right", True, output
    )

    print(output)
    print(output + starts[cp.newaxis, :].transpose())
