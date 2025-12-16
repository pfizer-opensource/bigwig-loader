import cupy as cp
import pytest

from bigwig_loader.searchsorted import interval_searchsorted
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


# ========== Tests for empty subarrays ==========


@pytest.fixture
def test_data_with_empty_subarrays():
    """Test data where some subarrays are empty (size=0)."""
    intervals_track1 = [5, 10, 12, 18]
    # track2 is empty
    intervals_track3 = [1, 2, 3, 4, 5]
    # track4 is empty
    intervals_track5 = [100, 200]

    array = cp.asarray(
        intervals_track1 + intervals_track3 + intervals_track5,
        dtype=cp.uint32,
    )
    queries = cp.asarray([7, 9, 11], dtype=cp.uint32)
    sizes = cp.asarray(
        [
            len(intervals_track1),  # 4
            0,  # empty
            len(intervals_track3),  # 5
            0,  # empty
            len(intervals_track5),  # 2
        ],
        dtype=cp.uint32,
    )
    return array, queries, sizes


def test_searchsorted_with_empty_subarrays_left_absolute(
    test_data_with_empty_subarrays,
) -> None:
    """Empty subarrays should return 0 (no valid indices)."""
    array, queries, sizes = test_data_with_empty_subarrays
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=True
    )

    # Track 0: [5, 10, 12, 18], queries [7, 9, 11] -> left indices [1, 1, 2] + start 0 = [1, 1, 2]
    # Track 1: empty -> [0, 0, 0]
    # Track 2: [1, 2, 3, 4, 5], queries [7, 9, 11] -> all past end -> [5, 5, 5] + start 4 = [9, 9, 9]
    # Track 3: empty -> [0, 0, 0]
    # Track 4: [100, 200], queries [7, 9, 11] -> all before start -> [0, 0, 0] + start 9 = [9, 9, 9]

    expected = cp.asarray(
        [
            [1, 1, 2],  # track 0
            [0, 0, 0],  # track 1 (empty)
            [9, 9, 9],  # track 2
            [0, 0, 0],  # track 3 (empty)
            [9, 9, 9],  # track 4
        ],
        dtype=cp.uint32,
    )

    assert (
        output == expected
    ).all(), f"Expected:\n{expected.get()}\nGot:\n{output.get()}"


def test_searchsorted_with_empty_subarrays_right_absolute(
    test_data_with_empty_subarrays,
) -> None:
    """Empty subarrays should return 0 (no valid indices)."""
    array, queries, sizes = test_data_with_empty_subarrays
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="right", absolute_indices=True
    )

    # Track 0: [5, 10, 12, 18], queries [7, 9, 11] -> right indices [1, 1, 2] + start 0 = [1, 1, 2]
    # Track 1: empty -> [0, 0, 0]
    # Track 2: [1, 2, 3, 4, 5], queries [7, 9, 11] -> all past end -> [5, 5, 5] + start 4 = [9, 9, 9]
    # Track 3: empty -> [0, 0, 0]
    # Track 4: [100, 200], queries [7, 9, 11] -> all before start -> [0, 0, 0] + start 9 = [9, 9, 9]

    expected = cp.asarray(
        [
            [1, 1, 2],  # track 0
            [0, 0, 0],  # track 1 (empty)
            [9, 9, 9],  # track 2
            [0, 0, 0],  # track 3 (empty)
            [9, 9, 9],  # track 4
        ],
        dtype=cp.uint32,
    )

    assert (
        output == expected
    ).all(), f"Expected:\n{expected.get()}\nGot:\n{output.get()}"


def test_searchsorted_with_empty_subarrays_relative(
    test_data_with_empty_subarrays,
) -> None:
    """Empty subarrays should return 0 in relative indices too."""
    array, queries, sizes = test_data_with_empty_subarrays
    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=False
    )

    # Track 1 and 3 are empty, should return 0
    assert (
        output[1, :] == 0
    ).all(), f"Empty track 1 should have all zeros, got {output[1].get()}"
    assert (
        output[3, :] == 0
    ).all(), f"Empty track 3 should have all zeros, got {output[3].get()}"


def test_searchsorted_all_empty_subarrays() -> None:
    """Test when ALL subarrays are empty."""
    array = cp.asarray([], dtype=cp.uint32)
    queries = cp.asarray([7, 9, 11], dtype=cp.uint32)
    sizes = cp.asarray([0, 0, 0], dtype=cp.uint32)

    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=True
    )

    expected = cp.zeros((3, 3), dtype=cp.uint32)
    assert (output == expected).all(), f"Expected all zeros, got:\n{output.get()}"


def test_searchsorted_single_empty_subarray() -> None:
    """Test with a single empty subarray."""
    array = cp.asarray([], dtype=cp.uint32)
    queries = cp.asarray([1, 2, 3], dtype=cp.uint32)
    sizes = cp.asarray([0], dtype=cp.uint32)

    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="right", absolute_indices=True
    )

    expected = cp.zeros((1, 3), dtype=cp.uint32)
    assert (output == expected).all()


def test_searchsorted_empty_subarray_at_start() -> None:
    """Test with empty subarray at the beginning."""
    array = cp.asarray([10, 20, 30], dtype=cp.uint32)
    queries = cp.asarray([15, 25], dtype=cp.uint32)
    sizes = cp.asarray([0, 3], dtype=cp.uint32)  # First empty, second has 3 elements

    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=True
    )

    # Track 0: empty -> [0, 0]
    # Track 1: [10, 20, 30], queries [15, 25] -> [1, 2] + start 0 = [1, 2]
    expected = cp.asarray(
        [
            [0, 0],  # empty track
            [1, 2],  # normal track
        ],
        dtype=cp.uint32,
    )

    assert (
        output == expected
    ).all(), f"Expected:\n{expected.get()}\nGot:\n{output.get()}"


def test_searchsorted_empty_subarray_at_end() -> None:
    """Test with empty subarray at the end."""
    array = cp.asarray([10, 20, 30], dtype=cp.uint32)
    queries = cp.asarray([15, 25], dtype=cp.uint32)
    sizes = cp.asarray([3, 0], dtype=cp.uint32)  # First has 3 elements, second empty

    output = searchsorted(
        array=array, queries=queries, sizes=sizes, side="left", absolute_indices=True
    )

    # Track 0: [10, 20, 30], queries [15, 25] -> [1, 2] + start 0 = [1, 2]
    # Track 1: empty -> [0, 0]
    expected = cp.asarray(
        [
            [1, 2],  # normal track
            [0, 0],  # empty track
        ],
        dtype=cp.uint32,
    )

    assert (
        output == expected
    ).all(), f"Expected:\n{expected.get()}\nGot:\n{output.get()}"


# ========== Tests for interval_searchsorted with empty subarrays ==========


def test_interval_searchsorted_with_empty_subarrays() -> None:
    """Test interval_searchsorted with some empty tracks."""
    # Intervals: track 0 has data, track 1 is empty, track 2 has data
    # Track 0: intervals [10-20], [30-40], [50-60]
    # Track 1: empty
    # Track 2: intervals [100-110], [120-130]

    array_start = cp.asarray([10, 30, 50, 100, 120], dtype=cp.uint32)
    array_end = cp.asarray([20, 40, 60, 110, 130], dtype=cp.uint32)

    query_starts = cp.asarray([35], dtype=cp.uint32)  # Query region 35-45
    query_ends = cp.asarray([45], dtype=cp.uint32)

    sizes = cp.asarray([3, 0, 2], dtype=cp.uint32)  # Track 1 is empty

    found_starts, found_ends = interval_searchsorted(
        array_start, array_end, query_starts, query_ends, sizes=sizes
    )

    # For track 1 (empty), found_starts and found_ends should both be 0
    # This ensures the interval loop does nothing
    assert (
        found_starts[1, 0] == 0
    ), f"Empty track found_starts should be 0, got {found_starts[1, 0]}"
    assert (
        found_ends[1, 0] == 0
    ), f"Empty track found_ends should be 0, got {found_ends[1, 0]}"

    # For empty tracks, found_starts should equal found_ends (empty range)
    assert (
        found_starts[1, 0] == found_ends[1, 0]
    ), "Empty track should have found_starts == found_ends"


def test_interval_searchsorted_empty_tracks_no_intervals_to_process() -> None:
    """Verify that empty tracks result in no intervals being selected."""
    # All empty tracks
    array_start = cp.asarray([], dtype=cp.uint32)
    array_end = cp.asarray([], dtype=cp.uint32)

    query_starts = cp.asarray([100, 200], dtype=cp.uint32)
    query_ends = cp.asarray([150, 250], dtype=cp.uint32)

    sizes = cp.asarray([0, 0, 0], dtype=cp.uint32)  # 3 empty tracks

    found_starts, found_ends = interval_searchsorted(
        array_start, array_end, query_starts, query_ends, sizes=sizes
    )

    # All should be zero
    assert (
        found_starts == 0
    ).all(), f"All found_starts should be 0, got:\n{found_starts.get()}"
    assert (
        found_ends == 0
    ).all(), f"All found_ends should be 0, got:\n{found_ends.get()}"

    # Verify shape is correct
    assert found_starts.shape == (
        3,
        2,
    ), f"Expected shape (3, 2), got {found_starts.shape}"
    assert found_ends.shape == (3, 2), f"Expected shape (3, 2), got {found_ends.shape}"


def test_interval_searchsorted_mixed_empty_and_nonempty() -> None:
    """Test with a realistic mix of empty and non-empty tracks."""
    # Simulating 5 tracks where tracks 1 and 3 have no data for this chromosome
    # Track 0: 2 intervals
    # Track 1: empty
    # Track 2: 3 intervals
    # Track 3: empty
    # Track 4: 1 interval

    array_start = cp.asarray([10, 50, 100, 150, 200, 300], dtype=cp.uint32)
    array_end = cp.asarray([20, 60, 120, 180, 220, 350], dtype=cp.uint32)

    query_starts = cp.asarray([15], dtype=cp.uint32)
    query_ends = cp.asarray([160], dtype=cp.uint32)

    sizes = cp.asarray([2, 0, 3, 0, 1], dtype=cp.uint32)

    found_starts, found_ends = interval_searchsorted(
        array_start, array_end, query_starts, query_ends, sizes=sizes
    )

    # Verify empty tracks have found_starts == found_ends == 0
    assert (
        found_starts[1, 0] == 0 and found_ends[1, 0] == 0
    ), "Track 1 (empty) should have zeros"
    assert (
        found_starts[3, 0] == 0 and found_ends[3, 0] == 0
    ), "Track 3 (empty) should have zeros"

    # Verify non-empty tracks have valid ranges (found_starts <= found_ends for non-empty results)
    # or found_starts > found_ends for regions with no overlapping intervals
    for i in [0, 2, 4]:
        start = int(found_starts[i, 0])
        end = int(found_ends[i, 0])
        # Either valid range or empty range, but not garbage values
        assert start < 1000000, f"Track {i} found_starts looks like garbage: {start}"
        assert end < 1000000, f"Track {i} found_ends looks like garbage: {end}"
