import numpy as np
import pandas as pd

from bigwig_loader.subtract_intervals import subtract_interval_dataframe


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


def test_subtract_intervals_overlapping_starts():
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
