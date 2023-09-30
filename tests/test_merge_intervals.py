import numpy as np
import pandas as pd

from bigwig_loader.merge_intervals import merge_interval_dataframe
from bigwig_loader.merge_intervals import merge_intervals
from bigwig_loader.merge_intervals import merge_intervals_with_chrom_keys


def test_merge_intervals() -> None:
    chrom_id = np.array([1, 1, 1, 1], np.dtype("uint32"))
    start = np.array([1, 4, 12, 16], np.dtype("uint32"))
    end = np.array([4, 9, 16, 19], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0], np.dtype("f4"))
    out_chrom_id, out_start, out_end, out_value = merge_intervals(
        chrom_id, start, end, value, True, 0
    )
    assert list(out_start) == [1, 12] and list(out_end) == [9, 19]


def test_merge_interval_2() -> None:
    chrom_id = np.array([1, 1, 1, 1, 1], np.dtype("uint32"))
    start = np.array([1, 4, 12, 16, 40], np.dtype("uint32"))
    end = np.array([4, 9, 16, 19, 45], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.dtype("f4"))
    out_chrom_id, out_start, out_end, out_value = merge_intervals(
        chrom_id, start, end, value, True, 0
    )
    assert list(out_start) == [1, 12, 40] and list(out_end) == [9, 19, 45]


def test_merge_interval_3() -> None:
    chrom_id = np.array([1, 1, 1, 1, 1], np.dtype("uint32"))
    start = np.array([1, 4, 12, 16, 40], np.dtype("uint32"))
    end = np.array([6, 9, 16, 19, 45], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.dtype("f4"))
    out_chrom_id, out_start, out_end, out_value = merge_intervals(
        chrom_id, start, end, value, True, 0
    )
    assert list(out_start) == [1, 12, 40] and list(out_end) == [9, 19, 45]


def test_merge_interval_4() -> None:
    chrom_id = np.array([1, 2, 3, 3, 3], np.dtype("uint32"))
    start = np.array([1, 4, 12, 16, 40], np.dtype("uint32"))
    end = np.array([6, 9, 16, 19, 45], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.dtype("f4"))
    out_chrom_id, out_start, out_end, out_value = merge_intervals(
        chrom_id, start, end, value, True, 0
    )
    assert (
        list(out_chrom_id) == [1, 2, 3, 3]
        and list(out_start) == [1, 4, 12, 40]
        and list(out_end) == [6, 9, 19, 45]
    )


def test_merge_interval_with_chrom_keys() -> None:
    chrom_id = np.array(["chr2", "chr4", "chr12", "chr12", "chr12"], dtype=object)
    start = np.array([1, 4, 12, 16, 40], np.dtype("uint32"))
    end = np.array([6, 9, 16, 19, 45], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.dtype("f4"))
    out_chrom_id, out_start, out_end, out_value = merge_intervals_with_chrom_keys(
        chrom_id, start, end, value, False, 0
    )
    assert (
        list(out_chrom_id) == ["chr2", "chr4", "chr12", "chr12"]
        and list(out_start) == [1, 4, 12, 40]
        and list(out_end) == [6, 9, 19, 45]
    )


def test_merge_intervals_dataframe() -> None:
    chrom = np.array(["chr2", "chr4", "chr12", "chr12", "chr12"])
    start = np.array([1, 4, 12, 16, 40], np.dtype("uint32"))
    end = np.array([6, 9, 16, 19, 45], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.dtype("f4"))

    df = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})
    df_shuffled = df.sample(frac=1)

    merged = merge_interval_dataframe(df_shuffled, False)
    assert list(merged["chrom"]) == ["chr2", "chr4", "chr12", "chr12"]
    assert list(merged["start"]) == [1, 4, 12, 40]
    assert list(merged["end"]) == [6, 9, 19, 45]


def test_merge_interval_with_allow_gap() -> None:
    chrom_id = np.array([1, 2, 3, 3, 3], np.dtype("uint32"))
    start = np.array([1, 4, 12, 20, 29], np.dtype("uint32"))
    end = np.array([6, 9, 16, 25, 32], np.dtype("uint32"))
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.dtype("f4"))
    out_chrom_id, out_start, out_end, out_value = merge_intervals(
        chrom_id, start, end, value, True, 4
    )
    assert (
        list(out_chrom_id) == [1, 2, 3]
        and list(out_start) == [1, 4, 12]
        and list(out_end) == [6, 9, 32]
    )
