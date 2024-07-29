import cupy as cp

from bigwig_loader.searchsorted import searchsorted


def test_searchsorted() -> None:
    array = cp.asarray(
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
    queries = cp.asarray([7, 9, 11], dtype=cp.int32)
    sizes = cp.asarray([4, 6, 14, 2], cp.int32)

    output = searchsorted(array=array, queries=queries, sizes=sizes, side="left")
    expected = cp.asarray([[1, 1, 2], [3, 4, 6], [6, 8, 10], [1, 1, 1]])
    assert (output == expected).all()
    output = searchsorted(array=array, queries=queries, sizes=sizes, side="right")
    expected = cp.asarray([[1, 1, 2], [4, 5, 6], [7, 9, 11], [1, 1, 1]])
    assert (output == expected).all()
