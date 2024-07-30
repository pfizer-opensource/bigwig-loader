import cupy as cp
import pytest

from bigwig_loader.searchsorted import searchsorted


@pytest.fixture
def test_data():
    intervals_track1 = [5, 10, 12, 18]
    intervals_track2 = [
        1,
        3,
        5,
        7,
        9,
        10,
    ]

    intervals_track3 = [
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
    ]

    intervals_track4 = [4, 100]

    array = cp.asarray(
        intervals_track1 + intervals_track2 + intervals_track3 + intervals_track4,
        dtype=cp.int32,
    )
    queries = cp.asarray([7, 9, 11], dtype=cp.int32)
    sizes = cp.asarray(
        [
            len(intervals_track1),
            len(intervals_track2),
            len(intervals_track3),
            len(intervals_track4),
        ],
        cp.int32,
    )
    return array, queries, sizes


def test_searchsorted_left_relative(test_data) -> None:
    array, queries, sizes = test_data
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=False
    )
    expected = cp.asarray([[1, 1, 2], [3, 4, 6], [6, 8, 10], [1, 1, 1]])
    assert (output == expected).all()


def test_searchsorted_right_relative(test_data) -> None:
    array, queries, sizes = test_data
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="right", absolute_indices=False
    )
    expected = cp.asarray([[1, 1, 2], [4, 5, 6], [7, 9, 11], [1, 1, 1]])
    assert (output == expected).all()


def test_searchsorted_left_absolute(test_data) -> None:
    array, queries, sizes = test_data
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=True
    )
    expected = cp.asarray([[1, 1, 2], [7, 8, 10], [16, 18, 20], [25, 25, 25]])
    assert (output == expected).all()


def test_searchsorted_right_absolute(test_data) -> None:
    array, queries, sizes = test_data
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="right", absolute_indices=True
    )
    expected = cp.asarray([[1, 1, 2], [8, 9, 10], [17, 19, 21], [25, 25, 25]])
    assert (output == expected).all()
