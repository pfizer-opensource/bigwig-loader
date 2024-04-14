import cupy as cp

from bigwig_loader.cupy_functions import moving_average


def test_correct_shape():
    array = cp.random.rand(10, 20, 30)
    result = moving_average(array, 8)
    print(result)
    assert result.shape == array.shape
