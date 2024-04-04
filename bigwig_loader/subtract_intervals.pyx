import numpy as np
import pandas as pd
from natsort import natsorted

cimport cython
cimport numpy as cnp

ctypedef cnp.npy_uint32 UINT32_t
ctypedef cnp.npy_uint8 UINT8_t

cdef list ID_TO_KEY_CANONICAL = [f"chr{x}" for x in range(1, 23)] + ["chrX", "chrY", "chrM"]
cdef set CANONICAL_CHROM_KEYS = set(ID_TO_KEY_CANONICAL)


def subtract_interval_dataframe(intervals: pd.DataFrame, blacklist: pd.DataFrame, buffer: int = 0) -> pd.DataFrame:
    """

    Args:
        intervals: basis for the intervals
        blacklist: intervals to subtract
        buffer: extra buffer zone around blacklist intervals

    Returns: pd.Dataframe with intervals not containing blacklisted
        intervals.

    """
    intervals = intervals.copy(deep=False)
    blacklist = blacklist.copy(deep=False)

    intervals["blacklist"] = False
    blacklist["blacklist"] = True
    blacklist["value"] = np.array([0.0], np.dtype("f4"))

    if buffer:
        blacklist["start"] = (blacklist["start"] - buffer).clip(lower=0)
        blacklist["end"] = blacklist["end"] + buffer

    combined = pd.concat([intervals, blacklist])

    chrom, start, end, value = subtract_intervals_with_chrom_keys(
                                                                combined["chrom"].values,
                                                                combined["start"].values,
                                                                combined["end"].values,
                                                                combined["value"].values,
                                                                combined["blacklist"].values
                                                               )

    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})



def subtract_intervals_with_chrom_keys(cnp.ndarray[object, ndim=1] chroms,
                                    cnp.ndarray[UINT32_t, ndim=1] starts,
                                    cnp.ndarray[UINT32_t, ndim=1] ends,
                                    cnp.ndarray[cnp.float32_t, ndim=1] values,
                                    cnp.ndarray[UINT8_t, ndim=1] blacklist):
    id_to_key = np.array(ID_TO_KEY_CANONICAL + natsorted(set(chroms) - CANONICAL_CHROM_KEYS), dtype=object)
    key_to_id = {key: i for i, key in enumerate(id_to_key)}
    chrom_ids = np.array([key_to_id[key] for key in chroms], dtype="uint32")
    chrom_ids_out, starts_out, ends_out, values_out = subtract_intervals(chrom_ids, starts, ends, values, blacklist)
    chroms_out = id_to_key[chrom_ids_out]
    return chroms_out, starts_out, ends_out, values_out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef subtract_intervals(cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_ids,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] starts,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] ends,
                      cnp.ndarray[cnp.float32_t, ndim = 1] values,
                      cnp.ndarray[cnp.uint8_t, ndim = 1] blacklist
                      ):
    cdef:
        size_t i
        size_t j
        size_t n_intervals
        UINT32_t chrom_id
        UINT32_t start
        UINT32_t end
        UINT8_t is_blacklist
        float value
        UINT32_t chrom_id_next
        UINT32_t start_next
        UINT32_t end_next
        UINT8_t is_blacklist_next
        UINT32_t first_allowed_start
        float zero = 0.0

    n_intervals = starts.shape[0]

    # cdef cnp.ndarray[cnp.int32_t, ndim = 1] idx = np.empty_like(chrom_ids, dtype="int32")

    cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_id_out = np.empty_like(chrom_ids)
    cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] start_out = np.empty_like(starts)
    cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] end_out = np.empty_like(ends)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] value_out = np.empty_like(values)

    cdef UINT32_t[:] chrom_id_out_view = chrom_id_out
    cdef UINT32_t[:] start_out_view = start_out
    cdef UINT32_t[:] end_out_view = end_out
    cdef float[:] value_out_view = value_out

    idx = np.lexsort([starts, chrom_ids])
    chrom_ids = chrom_ids[idx]
    starts = starts[idx]
    ends = ends[idx]
    values = values[idx]
    blacklist =  blacklist[idx]

    i = 1
    j = 0

    cdef UINT32_t[:] chrom_id_view = chrom_ids
    cdef UINT32_t[:] start_view = starts
    cdef UINT32_t[:] end_view = ends
    cdef float[:] value_view = values

    first_allowed_start = starts[0]
    # value_new = zero
    while i < n_intervals:
        chrom_id = chrom_id_view[i - 1]
        start = start_view[i - 1]
        end = end_view[i - 1]
        value = value_view[i - 1]
        is_blacklist = blacklist[i - 1]

        chrom_id_next = chrom_id_view[i]
        start_next = start_view[i]
        end_next = end_view[i]
        is_blacklist_next = blacklist[i]

        if is_blacklist:
            first_allowed_start = end
            i += 1
            continue
        if end <= first_allowed_start:
            i += 1
            continue
        if start == start_next:
            i += 1
            continue

        first_allowed_start = max(first_allowed_start, start)


        chrom_id_out_view[j] = chrom_id
        start_out_view[j] = first_allowed_start
        value_out_view[j] = value
        if is_blacklist_next and start_next < end and chrom_id == chrom_id_next:
            end_out_view[j] = min(end, start_next)
        else:
            end_out_view[j] = end
        j += 1
        i += 1

    return chrom_id_out[:j], start_out[:j], end_out[:j], value_out[:j]
