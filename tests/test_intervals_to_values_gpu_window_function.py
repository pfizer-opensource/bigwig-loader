import cupy as cp

from bigwig_loader.intervals_to_values_gpu import intervals_to_values


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

    print(expected)
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
