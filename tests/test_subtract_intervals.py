import random

import numpy as np
import pandas as pd

from bigwig_loader.subtract_intervals import subtract_interval_dataframe


def test_complete_overlap():
    """
    interval:     |-----|
    blacklist:    |-----|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == []
    assert list(result["start"]) == []
    assert list(result["end"]) == []


def test_same_start_longer_interval():
    """
    interval:     |-------|
    blacklist:    |-----|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([22], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [20]
    assert list(result["end"]) == [22]


def test_same_start_longer_blacklist():
    """
    interval:     |-----|
    blacklist:    |-------|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([22], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == []
    assert list(result["start"]) == []
    assert list(result["end"]) == []


def test_same_end_longer_blacklist():
    """
    interval:       |-----|
    blacklist:    |-------|
    """
    chrom = np.array(["chr2"])
    start = np.array([6], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == []
    assert list(result["start"]) == []
    assert list(result["end"]) == []


def test_same_end_longer_interval():
    """
    interval:     |-------|
    blacklist:      |-----|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([6], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [4]
    assert list(result["end"]) == [6]


def test_first_interval_then_blacklist():
    """
    interval:     |-------|
    blacklist:      |-------|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([6], np.dtype("uint32"))
    end = np.array([22], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [4]
    assert list(result["end"]) == [6]


def test_first_blacklist_then_interval():
    """
    interval:       |-------|
    blacklist:    |-------|

    HANGS!!
    """
    chrom = np.array(["chr2"])
    start = np.array([6], np.dtype("uint32"))
    end = np.array([22], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [20]
    assert list(result["end"]) == [22]


def test_interval_within_blacklist():
    """
    interval:       |-----|
    blacklist:    |---------|
    """
    chrom = np.array(["chr2"])
    start = np.array([6], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([22], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == []
    assert list(result["start"]) == []
    assert list(result["end"]) == []


def test_blacklist_within_interval():
    """
    interval:    |-----------|
    blacklist:      |-----|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([22], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([6], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print("result:", result)

    assert list(result["chrom"]) == ["chr2", "chr2"]
    assert list(result["start"]) == [4, 20]
    assert list(result["end"]) == [6, 22]


def test_subtract_intervals_no_overlap():
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([100], np.dtype("uint32"))
    end = np.array([200], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [4]
    assert list(result["end"]) == [20]


def test_subtract_intervals_dataframe():
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([8], np.dtype("uint32"))
    end = np.array([40], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [4]
    assert list(result["end"]) == [8]


def test_subtract_intervals_dataframe_different_chromosomes():
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr3"])
    start = np.array([8], np.dtype("uint32"))
    end = np.array([40], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [4]
    assert list(result["end"]) == [20]


def test_subtract_intervals_dataframe_different_chromosomes_switched():
    chrom = np.array(["chr3"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([8], np.dtype("uint32"))
    end = np.array([40], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    assert list(result["chrom"]) == ["chr3"]
    assert list(result["start"]) == [4]
    assert list(result["end"]) == [20]


def test_subtract_intervals_overlapping_starts():
    """
    interval:     |-----|
    blacklist:    |----------|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([40], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == []
    assert list(result["start"]) == []
    assert list(result["end"]) == []


def test_subtract_intervals_overlapping_starts_contained_blacklist():
    """
    interval:     |-------|
    blacklist:    |----|
    """
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([10], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == ["chr2"]
    assert list(result["start"]) == [10]
    assert list(result["end"]) == [20]


def test_subtract_intervals_dataframe_encapsulated_by_blacklist():
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([20], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([2], np.dtype("uint32"))
    end = np.array([40], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == []
    assert list(result["start"]) == []
    assert list(result["end"]) == []


def test_interval_should_be_splitted_by_blacklist():
    chrom = np.array(["chr2"])
    start = np.array([4], np.dtype("uint32"))
    end = np.array([100], np.dtype("uint32"))
    value = np.array([1.0], np.dtype("f4"))

    df1 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    chrom = np.array(["chr2"])
    start = np.array([10], np.dtype("uint32"))
    end = np.array([40], np.dtype("uint32"))

    df2 = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

    result = subtract_interval_dataframe(intervals=df1, blacklist=df2)

    print(result)

    assert list(result["chrom"]) == ["chr2", "chr2"]
    assert list(result["start"]) == [4, 40]
    assert list(result["end"]) == [10, 100]


def create_non_overlapping_sorted_intervals(n_intervals=500):
    interval_starts = []
    interval_ends = []
    start = 0
    for _ in range(n_intervals):
        start = start + random.randint(0, 50)
        end = start + random.randint(0, 50)
        interval_starts.append(start)
        interval_ends.append(end)
        start = end
    return pd.DataFrame(
        {
            "chrom": ["chr4"] * len(interval_starts),
            "start": np.array(interval_starts, np.dtype("uint32")),
            "end": np.array(interval_ends, np.dtype("uint32")),
            "value": np.array([1.0] * len(interval_starts), np.dtype("f4")),
        }
    )


def get_all_indexes_within_interval(dataframe):
    indeces = []
    for _, row in dataframe.iterrows():
        start = row["start"]
        end = row["end"]
        indeces.extend(range(start, end))
    return indeces


def test_randomly_generated_intervals_and_blacklist():
    # Base intervals should not be overlapping
    intervals = create_non_overlapping_sorted_intervals()
    # Blacklist intervals are allowed to overlap eachother
    blacklist = pd.concat(
        [
            create_non_overlapping_sorted_intervals(),
            create_non_overlapping_sorted_intervals(),
        ]
    )
    blacklist.sort_values(by="start", inplace=True)
    result = subtract_interval_dataframe(intervals=intervals, blacklist=blacklist)

    # Test by explicitely creating a set of all indexes within
    # intervals and blacklist and then subtracting the blacklist
    # from the intervals.
    result = set(get_all_indexes_within_interval(result))
    intervals = set(get_all_indexes_within_interval(intervals))
    blacklist = set(get_all_indexes_within_interval(blacklist))

    assert result == intervals - blacklist
