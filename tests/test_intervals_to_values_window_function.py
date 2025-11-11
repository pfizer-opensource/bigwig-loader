from itertools import product
from math import isnan

import cupy as cp
import numpy as np
import pytest

from bigwig_loader.intervals_to_values import intervals_to_values


def reshape_expected(
    expected: cp.ndarray, batch_size: int, n_tracks: int
) -> cp.ndarray:
    """
    Reshape expected output to match new shape (batch_size, sequence_length, n_tracks).

    Handles cases where expected might be:
    - 1D: sequence_length → reshape to (batch_size, sequence_length, n_tracks)
    - 2D: (batch_size, sequence_length) or (n_tracks, sequence_length) → reshape appropriately
    - 3D: already correct or needs transposing
    """
    if expected.ndim == 1:
        # 1D array: expand to (batch_size, sequence_length, n_tracks)
        seq_length = len(expected)
        expected = expected.reshape(1, seq_length, 1)
        if batch_size > 1 or n_tracks > 1:
            expected = cp.broadcast_to(expected, (batch_size, seq_length, n_tracks))
    elif expected.ndim == 2:
        # 2D array: assume (batch_size, sequence_length), add track dimension
        expected = expected.reshape(expected.shape[0], expected.shape[1], 1)
        if n_tracks > 1:
            expected = cp.broadcast_to(
                expected, (batch_size, expected.shape[1], n_tracks)
            )
    elif expected.ndim == 3:
        # Already 3D: check if it needs transposing or reshaping
        # Case 1: shape is (n_tracks, batch_size, sequence_length), transpose to (batch_size, sequence_length, n_tracks)
        if expected.shape[0] == n_tracks and expected.shape[1] == batch_size:
            expected = expected.transpose(1, 2, 0)
        # Case 2: shape is (batch_size, 1, sequence_length), reshape to (batch_size, sequence_length, n_tracks)
        elif expected.shape[0] == batch_size and expected.shape[1] == 1:
            seq_length = expected.shape[2]
            expected = expected.reshape(batch_size, seq_length, 1)
            if n_tracks > 1:
                expected = cp.broadcast_to(expected, (batch_size, seq_length, n_tracks))

    return expected


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_window(default_value) -> None:
    """."""
    track_starts = cp.asarray([1, 3, 10, 12, 16], dtype=cp.int32)
    track_ends = cp.asarray([3, 10, 12, 16, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 15.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([2], dtype=cp.int32)
    query_ends = cp.asarray([17], dtype=cp.int32)
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros((batch_size, 3, n_tracks), dtype=cp.float32)
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        window_size=5,
        default_value=default_value,
        out=reserved,
    )

    expected = cp.asarray([[16.0, 21.0, 42.0]])
    expected = reshape_expected(expected, batch_size, n_tracks)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert (values == expected).all()


@pytest.mark.parametrize("default_value", [0.0, cp.nan, 5.6, 10.0, 7565])
def test_get_values_from_intervals_edge_case_1(default_value) -> None:
    """Query start is somewhere in a "gap"."""
    track_starts = cp.asarray([1, 10, 12, 16], dtype=cp.int32)
    track_ends = cp.asarray([3, 12, 16, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([6], dtype=cp.int32)
    query_ends = cp.asarray([18], dtype=cp.int32)
    batch_size = len(query_starts)
    n_tracks = 1
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        default_value=default_value,
        window_size=3,
    )
    x = default_value
    if isnan(default_value):
        expected = cp.asarray([[x, 30.0, 40.0, 46.666668]])
    elif default_value != 0:
        expected = cp.asarray([[x, (x + 30.0 + 30.0) / 3, 40.0, 46.666668]])
    else:
        expected = cp.asarray([[x, 20.0, 40.0, 46.666668]])

    expected = reshape_expected(expected, batch_size, n_tracks)
    print("expected:")
    print(expected)
    print("actual:")
    print(values)

    assert cp.allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_edge_case_2(default_value) -> None:
    """Query start is exactly at start index after "gap"."""
    track_starts = cp.asarray([1, 10, 12, 16], dtype=cp.int32)
    track_ends = cp.asarray([3, 12, 16, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([10], dtype=cp.int32)
    query_ends = cp.asarray([18], dtype=cp.int32)
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros((batch_size, 2, n_tracks), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        window_size=4,
        default_value=default_value,
        out=reserved,
    )
    expected = cp.asarray([[35.0, 45.0]])
    expected = reshape_expected(expected, batch_size, n_tracks)
    print(expected)
    print(values)
    assert (
        cp.allclose(values, expected, equal_nan=True)
        and expected.shape[1] == (query_ends[0] - query_starts[0]) / 4
    )


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_edge_case_3(default_value) -> None:
    """Query end is somewhere in a "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([8], dtype=cp.int32)
    query_ends = cp.asarray([16], dtype=cp.int32)
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros((batch_size, 2, n_tracks), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        window_size=4,
        default_value=default_value,
        out=reserved,
    )
    if isnan(default_value):
        expected = cp.asarray([[25.0, 40.0]])
    else:
        expected = cp.asarray([[25.0, 20.0]])
    expected = reshape_expected(expected, batch_size, n_tracks)
    print(expected)
    print(values)
    assert cp.allclose(expected, values, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_edge_case_4(default_value) -> None:
    """Query end is exactly at end index before "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([8], dtype=cp.int32)
    query_ends = cp.asarray([14], dtype=cp.int32)
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros((batch_size, 2, n_tracks), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        window_size=3,
        default_value=default_value,
        out=reserved,
    )
    expected = cp.asarray([[23.333334, 36.666668]])
    expected = reshape_expected(expected, batch_size, n_tracks)

    print(expected)
    print(values)

    assert cp.allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_edge_case_5(default_value) -> None:
    """Query end is exactly at end index before "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.uint32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.uint32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([8], dtype=cp.uint32)
    query_ends = cp.asarray([20], dtype=cp.uint32)
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros((batch_size, 4, n_tracks), dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        window_size=3,
        default_value=default_value,
        out=reserved,
    )
    x = default_value
    if isnan(default_value):
        expected = cp.asarray([[23.333334, 36.666668, x, 50.0]])
    else:
        expected = cp.asarray([[23.333334, 36.666668, x, 33.333332]])

    expected = reshape_expected(expected, batch_size, n_tracks)
    print(expected)
    print(values)

    assert cp.allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_batch_of_2(default_value) -> None:
    """Query end is exactly at end index before "gap"."""
    track_starts = cp.asarray([5, 10, 12, 18], dtype=cp.int32)
    track_ends = cp.asarray([10, 12, 14, 20], dtype=cp.int32)
    track_values = cp.asarray([20.0, 30.0, 40.0, 50.0], dtype=cp.dtype("f4"))
    query_starts = cp.asarray([6, 8], dtype=cp.int32)
    query_ends = cp.asarray([18, 20], dtype=cp.int32)
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros([batch_size, 4, n_tracks], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        window_size=3,
        default_value=default_value,
        out=reserved,
    )

    if isnan(default_value):
        expected = cp.asarray(
            [[[20.0, 26.666666, 40.0, cp.nan]], [[23.333334, 36.666668, cp.nan, 50.0]]]
        )
    else:
        expected = cp.asarray(
            [
                [[20.0, 26.666666, 26.666666, 0.0]],
                [[23.333334, 36.666668, 0.0, 33.333332]],
            ]
        )

    expected = reshape_expected(expected, batch_size, n_tracks)
    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert cp.allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_one_track_one_sample(default_value) -> None:
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
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros([batch_size, 3, n_tracks], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=cp.asarray([4], dtype=cp.int32),
        window_size=3,
        default_value=default_value,
        out=reserved,
    )

    x = default_value
    expected = cp.asarray(
        [
            [
                [20.0, 30.0, 30.0, 40.0, 40.0, x, x, x, x, 50.0, 50.0],
            ],
        ]
    )

    def apply_window(full_matrix):
        return cp.stack(
            [
                cp.nanmean(full_matrix[:, :, :3], axis=2),
                cp.nanmean(full_matrix[:, :, 3:6], axis=2),
                cp.nanmean(full_matrix[:, :, 6:9], axis=2),
            ],
            axis=-1,
        )

    expected = apply_window(expected)
    expected = reshape_expected(expected, batch_size, n_tracks)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert cp.allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_one_track_one_sample_2(default_value) -> None:
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
    batch_size = len(query_starts)
    n_tracks = 1
    reserved = cp.zeros([batch_size, 3, n_tracks], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=cp.asarray([3], dtype=cp.int32),
        window_size=3,
        default_value=default_value,
        out=reserved,
    )
    x = default_value
    expected = cp.asarray(
        [
            [
                [
                    x,
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
                cp.nanmean(full_matrix[:, :, :3], axis=2),
                cp.nanmean(full_matrix[:, :, 3:6], axis=2),
                cp.nanmean(full_matrix[:, :, 6:9], axis=2),
            ],
            axis=-1,
        )

    expected = apply_window(expected)
    expected = reshape_expected(expected, batch_size, n_tracks)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    print("difference")
    print(values - expected)
    assert cp.allclose(values, expected, atol=1e-2, rtol=1e-2, equal_nan=True)


@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_get_values_from_intervals_multiple_tracks(default_value) -> None:
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
    batch_size = len(query_starts)
    n_tracks = 3
    reserved = cp.zeros([batch_size, 3, n_tracks], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=cp.asarray([4, 5, 3], dtype=cp.int32),
        window_size=3,
        default_value=default_value,
        out=reserved,
    )

    x = default_value
    expected = cp.asarray(
        [
            [
                [20.0, 30.0, 30.0, 40.0, 40.0, x, x, x, x, 50.0, 50.0],
            ],
            [
                [70.0, 80.0, 80.0, 80.0, 80.0, x, x, x, x, 90.0, 90.0],
            ],
            [
                [
                    x,
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
                cp.nanmean(full_matrix[:, :, :3], axis=2),
                cp.nanmean(full_matrix[:, :, 3:6], axis=2),
                cp.nanmean(full_matrix[:, :, 6:9], axis=2),
            ],
            axis=-1,
        )

    expected = apply_window(expected)
    expected = reshape_expected(expected, batch_size, n_tracks)

    print("expected:", expected.shape)
    print(expected)
    print("actual:", values.shape)
    print(values)
    print("difference")
    print(values - expected)
    assert cp.allclose(values, expected, atol=1e-2, rtol=1e-2, equal_nan=True)


@pytest.mark.parametrize(
    "sequence_length, window_size", product([8, 9, 10, 11, 13, 23], [2, 3, 4, 10, 11])
)
@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_combinations_sequence_length_window_size(
    sequence_length, window_size, default_value
) -> None:
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
    batch_size = len(query_starts)
    n_tracks = 3

    reserved = cp.zeros([batch_size, reduced_dim, n_tracks], dtype=cp.dtype("<f4"))
    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=cp.asarray([4, 5, 3], dtype=cp.int32),
        window_size=window_size,
        default_value=default_value,
        out=reserved,
    )

    reserved = cp.zeros([batch_size, sequence_length, n_tracks], dtype=cp.dtype("<f4"))
    values_with_window_size_1 = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=cp.asarray([4, 5, 3], dtype=cp.int32),
        window_size=1,
        default_value=default_value,
        out=reserved,
    )

    full_matrix = values_with_window_size_1[:, : reduced_dim * window_size, :]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], reduced_dim, window_size, full_matrix.shape[2]
    )
    expected = cp.nanmean(full_matrix, axis=2)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)
    assert cp.allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize(
    "window_size, batch_size, n_tracks", product([1, 2, 3], [1, 2, 3], [1, 2, 3])
)
@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_combinations_window_size_batch_size_n_tracks(
    window_size, batch_size, n_tracks, default_value
) -> None:
    """."""
    track_starts = cp.asarray([1, 3, 10, 12, 16] * n_tracks, dtype=cp.int32)
    track_ends = cp.asarray([3, 8, 12, 16, 20] * n_tracks, dtype=cp.int32)
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
        default_value=default_value,
    )

    values_with_window_size_1 = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=sizes,
        window_size=1,
        default_value=default_value,
    )

    reduced_dim = sequence_length // window_size
    full_matrix = values_with_window_size_1[:, : reduced_dim * window_size, :]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], reduced_dim, window_size, full_matrix.shape[2]
    )
    expected = cp.nanmean(full_matrix, axis=2)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)

    assert cp.allclose(values, expected, equal_nan=True)


def create_random_track_data(n_tracks, min_intervals=10, max_intervals=20):
    """Create random track data for testing."""
    track_starts = []
    track_ends = []
    values = []
    sizes = []
    for _ in range(n_tracks):
        current_start = 0
        generate_n_intervals = np.random.randint(min_intervals, max_intervals)
        for i in range(generate_n_intervals):
            start = current_start + np.random.randint(
                1, 50
            )  # Ensure a gap between intervals
            end = start + np.random.randint(1, 100)  # Random interval length
            track_starts.append(start)
            track_ends.append(end)
            values.append(np.random.random())
            current_start = end  # Update the start for the next interval
        sizes.append(generate_n_intervals)

    return (
        cp.asarray(track_starts, dtype=cp.int32),
        cp.asarray(track_ends, dtype=cp.int32),
        cp.asarray(values, dtype=cp.float32),
        cp.asarray(sizes, dtype=cp.int32),
    )


def create_random_queries(batch_size, sequence_length=200):
    start = np.random.randint(1, 50, size=batch_size)
    end = start + sequence_length
    return cp.asarray(start, dtype=cp.int32), cp.asarray(end, dtype=cp.int32)


@pytest.mark.parametrize(
    "window_size, batch_size, n_tracks",
    product([1, 2, 3, 7], [1, 2, 3, 7], [1, 2, 3, 7]),
)
@pytest.mark.parametrize("default_value", [0.0, cp.nan])
def test_combinations_window_size_batch_size_n_tracks_on_random_data(
    window_size, batch_size, n_tracks, default_value
) -> None:
    sequence_length = 200
    track_starts, track_ends, track_values, sizes = create_random_track_data(n_tracks)
    query_starts, query_ends = create_random_queries(
        batch_size, sequence_length=sequence_length
    )

    values = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=sizes,
        window_size=window_size,
        default_value=default_value,
    )

    values_with_window_size_1 = intervals_to_values(
        track_starts,
        track_ends,
        track_values,
        query_starts,
        query_ends,
        sizes=sizes,
        window_size=1,
        default_value=cp.nan,
    )

    cp.nan_to_num(values_with_window_size_1, copy=False, nan=default_value)

    reduced_dim = sequence_length // window_size
    full_matrix = values_with_window_size_1[:, : reduced_dim * window_size, :]
    full_matrix = full_matrix.reshape(
        full_matrix.shape[0], reduced_dim, window_size, full_matrix.shape[2]
    )
    expected = cp.nanmean(full_matrix, axis=2)

    print("expected:")
    print(expected)
    print("actual:")
    print(values)

    assert cp.allclose(values, expected, equal_nan=True)


if __name__ == "__main__":
    print(create_random_track_data(3))
