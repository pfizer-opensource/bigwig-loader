import cupy as cp


def moving_average(array: cp.ndarray, window_size: int) -> cp.ndarray:
    if window_size == 1:
        return array
    kernel = cp.ones(window_size) / window_size
    # pad_size = window_size // 2
    # arr_padded = cp.pad(array, ((0, 0), (0, 0), (pad_size, pad_size)), mode="constant")
    result = cp.apply_along_axis(
        lambda m: cp.convolve(m, kernel, mode="same"), axis=-1, arr=array
    )
    return result
