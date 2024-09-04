from itertools import product

import cupy as cp
import pytest

from bigwig_loader.intervals_to_values import intervals_to_values


def test_get_values_from_intervals_window() -> None:
    """."""
    track_starts = cp.asarray([1, 3, 10, 12, 16], dtype=cp.int32)
    track_ends = cp.asarray([3, 10, 12, 16, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 15.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([2], dtype=cp.int32)
    query_ends = cp.asarray([17], dtype=cp.int32)
    reserved = cp.zeros((1, 3), dtype=cp.float32)
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=5,
    )

    expected = cp.asarray([[16.0, 21.0, 42.0]])

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert (values == expected).all()


def test_get_values_from_intervals_edge_case_1() -> None:
    """Query start is somewhere in a "gap"."""
    track_starts = cp.asarray([1, 10, 12, 16], dtype=cp.int32)
    track_ends = cp.asarray([3, 12, 16, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([6], dtype=cp.int32)
    query_ends = cp.asarray([18], dtype=cp.int32)
    reserved = cp.zeros((1, 4), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=3,
    )
    expected = cp.asarray([[0.0, 20.0, 40.0, 46.666668]])

    print(expected)
    print(values)

    assert (
        cp.allclose(values, expected)
        and expected.shape[-1] == (query_ends[0] - query_starts[0]) / 3
    )


def test_get_values_from_intervals_edge_case_2() -> None:
    """Query start is exactly at start index after "gap"."""
    track_starts = cp.asarray([1, 10, 12, 16], dtype=cp.int32)
    track_ends = cp.asarray([3, 12, 16, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([10], dtype=cp.int32)
    query_ends = cp.asarray([18], dtype=cp.int32)
    reserved = cp.zeros((1, 2), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=4,
    )
    expected = cp.asarray([[35.0, 45.0]])
    assert (
        cp.allclose(values, expected)
        and expected.shape[-1] == (query_ends[0] - query_starts[0]) / 4
    )


def test_get_values_from_intervals_edge_case_3() -> None:
    """Query end is somewhere in a "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([8], dtype=cp.int32)
    query_ends = cp.asarray([16], dtype=cp.int32)
    reserved = cp.zeros((1, 2), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=4,
    )
    # expected = cp.asarray([[20, 20, 30, 30, 40, 40, 0, 0]])
    expected = cp.asarray([[25.0, 20.0]])

    assert (values == expected).all() and expected.shape[-1] == (
        query_ends[0] - query_starts[0]
    ) / 4


def test_get_values_from_intervals_edge_case_4() -> None:
    """Query end is exactly at end index before "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([8], dtype=cp.int32)
    query_ends = cp.asarray([14], dtype=cp.int32)
    reserved = cp.zeros((1, 2), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=3,
    )
    # without window function:[[20, 20, 30, 30, 40, 40]]
    expected = cp.asarray([[23.333334, 36.666668]])

    print(expected)
    print(values)

    assert (
        cp.allclose(values, expected)
        and expected.shape[-1] == (query_ends[0] - query_starts[0]) / 3
    )


def test_get_values_from_intervals_edge_case_5() -> None:
    """Query end is exactly at end index before "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.uint32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.uint32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([8], dtype=cp.uint32)
    query_ends = cp.asarray([20], dtype=cp.uint32)
    reserved = cp.zeros((1, 4), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=3,
    )
    expected = cp.asarray([[23.333334, 36.666668, 0.0, 33.333332]])

    print(expected)
    print(values)

    assert (
        cp.allclose(values, expected)
        and expected.shape[-1] == (query_ends[0] - query_starts[0]) / 3
    )


def test_get_values_from_intervals_batch_of_2() -> None:
    """Query end is exactly at end index before "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([6, 8], dtype=cp.int32)
    query_ends = cp.asarray([18, 20], dtype=cp.int32)
    reserved = cp.zeros([2, 4], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        window_size=3,
    )
    # expected without window function:
    #    [20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0]
    #    [20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0]

    expected = cp.asarray(
        [[20.0, 26.666666, 26.666666, 0.0], [23.333334, 36.666668, 0.0, 33.333332]]
    )
    print(expected)
    print(values)
    assert cp.allclose(values, expected)


def test_one_track_one_sample() -> None:
    """
    This tests a specific combination of track and batch index
    of the larger test case below:
    test_get_values_from_intervals_batch_multiple_tracks
        track index = 0
        batch_index = 1
    Included to narrow down source of error in the larger test case.
    """
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray(
        [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
        dtype=cp.dtype("f4"),
    )
    query_starts = cp.asarray([9], dtype=cp.int32)
    query_ends = cp.asarray([20], dtype=cp.int32)
    reserved = cp.zeros([1, 1, 3], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        sizes=cp.asarray([4], dtype=cp.int32),
        window_size=3,
    )

    expected = cp.asarray(
        [
            [
                [20.0, 30.0, 30.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0],
            ],
        ]
    )

    def apply_window(full_matrix):
        return cp.stack(
            [
                cp.mean(full_matrix[:, :, :3], axis=2),
                cp.mean(full_matrix[:, :, 3:6], axis=2),
                cp.mean(full_matrix[:, :, 6:9], axis=2),
            ],
            axis=-1,
        )

    expected = apply_window(expected)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert cp.allclose(values, expected)


def test_one_track_one_sample_2() -> None:
    """
    This tests a specific combination of track and batch index
    of the larger test case below:
    test_get_values_from_intervals_batch_multiple_tracks
        track index = 2
        batch_index = 1
    Included to narrow down source of error in the larger test case.
    """
    track_starts = cp.asarray([10, 100, 1000], dtype=cp.int32)
    track_ends = cp.asarray([20, 200, 2000], dtype=cp.int32)
    track_values = cp.asarray(
        [110.0, 120.0, 130.0],
        dtype=cp.dtype("f4"),
    )
    query_starts = cp.asarray([9], dtype=cp.int32)
    query_ends = cp.asarray([20], dtype=cp.int32)
    reserved = cp.zeros([1, 1, 3], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        sizes=cp.asarray([3], dtype=cp.int32),
        window_size=3,
    )

    expected = cp.asarray(
        [
            [
                [
                    0.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                ],
            ]
        ]
    )

    def apply_window(full_matrix):
        return cp.stack(
            [
                cp.mean(full_matrix[:, :, :3], axis=2),
                cp.mean(full_matrix[:, :, 3:6], axis=2),
                cp.mean(full_matrix[:, :, 6:9], axis=2),
            ],
            axis=-1,
        )

    expected = apply_window(expected)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    print("difference")
    print(values - expected)
    assert cp.allclose(values, expected, atol=1e-2, rtol=1e-2)


def test_get_values_from_intervals_multiple_tracks() -> None:
    """Test interval_to_values with 3 tracks and a batch size of 1."""
    track_starts = cp.asarray(
        [5, 10, 12, 18, 8, 9, 10, 18, 25, 10, 100, 1000], dtype=cp.int32
    )
    track_ends = cp.asarray(
        [10, 12, 14, 20, 9, 10, 14, 22, 55, 20, 200, 2000], dtype=cp.int32
    )
    track_values = cp.asarray(
        [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
        dtype=cp.dtype("f4"),
    )
    query_starts = cp.asarray([9], dtype=cp.int32)
    query_ends = cp.asarray([20], dtype=cp.int32)
    reserved = cp.zeros([3, 1, 3], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        sizes=cp.asarray([4, 5, 3], dtype=cp.int32),
        window_size=3,
    )

    expected = cp.asarray(
        [
            [
                [20.0, 30.0, 30.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0],
            ],
            [
                [70.0, 80.0, 80.0, 80.0, 80.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0],
            ],
            [
                [
                    0.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                    110.0,
                ],
            ],
        ]
    )

    def apply_window(full_matrix):
        return cp.stack(
            [
                cp.mean(full_matrix[:, :, :3], axis=2),
                cp.mean(full_matrix[:, :, 3:6], axis=2),
                cp.mean(full_matrix[:, :, 6:9], axis=2),
            ],
            axis=-1,
        )

    expected = apply_window(expected)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    print("difference")
    print(values - expected)
    assert cp.allclose(values, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "sequence_length, window_size", product([8, 9, 10, 11, 13, 23], [2, 3, 4, 10, 11])
)
def test_combinations_sequence_length_window_size(sequence_length, window_size) -> None:
    """Test intervals_to_values with 3 tracks and a batch size of 4."""
    track_starts = cp.asarray(
        [5, 10, 12, 18, 8, 9, 10, 18, 25, 10, 100, 1000], dtype=cp.int32
    )
    track_ends = cp.asarray(
        [10, 12, 14, 20, 9, 10, 14, 22, 55, 20, 200, 2000], dtype=cp.int32
    )
    track_values = cp.asarray(
        [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
        dtype=cp.dtype("f4"),
    )
    query_starts = cp.asarray([7, 9, 20, 99], dtype=cp.int32)
    query_ends = query_starts + sequence_length
    reduced_dim = sequence_length // window_size

    reserved = cp.zeros([3, 4, reduced_dim], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        sizes=cp.asarray([4, 5, 3], dtype=cp.int32),
        window_size=window_size,
    )

    reserved = cp.zeros([3, 4, sequence_length], dtype=cp.dtype("<f4"))
    values_with_window_size_1 = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        reserved,
        sizes=cp.asarray([4, 5, 3], dtype=cp.int32),
        window_size=1,
    )

    full_matrix = values_with_window_size_1[:, :, : reduced_dim * window_size]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], full_matrix.shape[1], reduced_dim, window_size
    )
    expected = cp.mean(full_matrix, axis=-1)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert cp.allclose(values, expected)


@pytest.mark.parametrize(
    "window_size, batch_size, n_tracks", product([1, 2, 3], [1, 2, 3], [1, 2, 3])
)
def test_combinations_window_size_batch_size_n_tracks(
    window_size, batch_size, n_tracks
) -> None:
    """."""
    track_starts = cp.asarray([1, 3, 10, 12, 16] * n_tracks, dtype=cp.int32)
    track_ends = cp.asarray([3, 10, 12, 16, 20] * n_tracks, dtype=cp.int32)
    track_values = cp.asarray(
        [20.0, 15.0, 30.0, 40.0, 50.0] * n_tracks, dtype=cp.dtype("f4")
    )
    sizes = cp.asarray([5] * n_tracks, dtype=cp.int32)
    sequence_length = 15
    query_starts = cp.asarray([2] * batch_size, dtype=cp.int32)
    query_ends = query_starts + sequence_length
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=sizes,
        window_size=window_size,
    )

    values_with_window_size_1 = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=sizes,
        window_size=1,
    )

    reduced_dim = sequence_length // window_size
    full_matrix = values_with_window_size_1[:, :, : reduced_dim * window_size]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], full_matrix.shape[1], reduced_dim, window_size
    )
    expected = cp.mean(full_matrix, axis=-1)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)

    assert cp.allclose(values, expected)
