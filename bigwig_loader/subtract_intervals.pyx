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

    if buffer:
        blacklist["start"] = (blacklist["start"] - buffer).clip(lower=0)
        blacklist["end"] = blacklist["end"] + buffer

    chrom, start, end, value = subtract_intervals_with_chrom_keys(
                                                                intervals["chrom"].values,
                                                                intervals["start"].values,
                                                                intervals["end"].values,
                                                                intervals["value"].values,
                                                                blacklist["chrom"].values,
                                                                blacklist["start"].values,
                                                                blacklist["end"].values,
    )

    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})



def subtract_intervals_with_chrom_keys(cnp.ndarray[object, ndim=1] chroms,
                                    cnp.ndarray[UINT32_t, ndim=1] starts,
                                    cnp.ndarray[UINT32_t, ndim=1] ends,
                                    cnp.ndarray[cnp.float32_t, ndim=1] values,
                                    cnp.ndarray[object, ndim=1] chroms_blacklist,
                                    cnp.ndarray[UINT32_t, ndim=1] starts_blacklist,
                                    cnp.ndarray[UINT32_t, ndim=1] ends_blacklist):
    id_to_key = np.array(ID_TO_KEY_CANONICAL + natsorted(set(chroms) - CANONICAL_CHROM_KEYS), dtype=object)
    print(id_to_key)
    key_to_id = {key: i for i, key in enumerate(id_to_key)}
    chrom_ids = np.array([key_to_id[key] for key in chroms], dtype="uint32")
    chrom_ids_blacklist = np.array([key_to_id[key] for key in chroms_blacklist], dtype="uint32")
    chrom_ids_out, starts_out, ends_out, values_out = subtract_intervals(chrom_ids, starts, ends, values, chrom_ids_blacklist, starts_blacklist, ends_blacklist)
    chroms_out = id_to_key[chrom_ids_out]
    return chroms_out, starts_out, ends_out, values_out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef subtract_intervals(cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_ids,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] starts,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] ends,
                      cnp.ndarray[cnp.float32_t, ndim = 1] values,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_ids_blacklist,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] starts_blacklist,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] ends_blacklist,
                      ):
    cdef:
        size_t i
        size_t j
        size_t k
        size_t n_intervals
        size_t n_blacklist
        UINT32_t interval_chrom_id
        UINT32_t interval_start
        UINT32_t interval_end
        float interval_value
        UINT32_t blacklist_chrom_id
        UINT32_t blacklist_start
        UINT32_t blacklist_end
        # UINT32_t first_allowed_start
        float zero = 0.0
        bint increment_interval
        bint increment_blacklist

    n_intervals = starts.shape[0]
    n_blacklist = starts_blacklist.shape[0]

    # cdef cnp.ndarray[cnp.int32_t, ndim = 1] idx = np.empty_like(chrom_ids, dtype="int32")

    # cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_id_out = np.empty_like(chrom_ids)
    # cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] start_out = np.empty_like(starts)
    # cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] end_out = np.empty_like(ends)
    # cdef cnp.ndarray[cnp.float32_t, ndim = 1] value_out = np.empty_like(values)
    cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_id_out = np.empty(shape=(n_intervals + n_blacklist,), dtype=np.uint32)
    cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] start_out = np.empty(shape=(n_intervals + n_blacklist,), dtype=np.uint32)
    cdef cnp.ndarray[cnp.npy_uint32, ndim = 1] end_out = np.empty(shape=(n_intervals + n_blacklist,), dtype=np.uint32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] value_out = np.empty(shape=(n_intervals + n_blacklist,), dtype=np.float32)

    cdef UINT32_t[:] chrom_id_out_view = chrom_id_out
    cdef UINT32_t[:] start_out_view = start_out
    cdef UINT32_t[:] end_out_view = end_out
    cdef float[:] value_out_view = value_out

    idx = np.lexsort([starts, chrom_ids])
    chrom_ids = chrom_ids[idx]
    starts = starts[idx]
    ends = ends[idx]
    values = values[idx]

    idx = np.lexsort([starts_blacklist, chrom_ids_blacklist])
    chrom_ids_blacklist = chrom_ids_blacklist[idx]
    starts_blacklist = starts_blacklist[idx]
    ends_blacklist = ends_blacklist[idx]

    i = 0
    j = 0
    k = 0

    cdef UINT32_t[:] chrom_id_view = chrom_ids
    cdef UINT32_t[:] start_view = starts
    cdef UINT32_t[:] end_view = ends
    cdef float[:] value_view = values

    cdef UINT32_t[:] chrom_id_view_blacklist = chrom_ids_blacklist
    cdef UINT32_t[:] start_view_blacklist = starts_blacklist
    cdef UINT32_t[:] end_view_blacklist = ends_blacklist

    #Starting off with the first interval and blacklist
    interval_chrom_id = chrom_id_view[i]
    interval_start = start_view[i]
    interval_end = end_view[i]
    interval_value = value_view[i]
    increment_interval = False

    blacklist_chrom_id = chrom_id_view_blacklist[j]
    blacklist_start = start_view_blacklist[j]
    blacklist_end = end_view_blacklist[j]
    increment_blacklist = False

    # value_new = zero
    while True:
        if increment_interval:
            if i + 1 == n_intervals:
                break
            i += 1
            interval_chrom_id = chrom_id_view[i]
            interval_start = start_view[i]
            interval_end = end_view[i]
            interval_value = value_view[i]
            increment_interval = False

        if increment_blacklist:
            j += 1
            if j < n_blacklist:
                blacklist_chrom_id = chrom_id_view_blacklist[j]
                blacklist_start = start_view_blacklist[j]
                blacklist_end = end_view_blacklist[j]
                increment_blacklist = False

        if smaller(interval_chrom_id, interval_end, blacklist_chrom_id, blacklist_start) or j >= n_blacklist:

            chrom_id_out_view[k] = interval_chrom_id
            start_out_view[k] = interval_start
            end_out_view[k] = interval_end
            value_out_view[k] = interval_value
            k += 1
            #result.append(intervals[i])
            increment_interval = True

        #elif interval_start > blacklist_end:
        elif larger(interval_chrom_id,  interval_start, blacklist_chrom_id, blacklist_end):
            increment_blacklist = True

        # interval is completely blacklisted
        #elif interval_start >= blacklist_start and interval_end <= blacklist_end:
        elif larger_equal(interval_chrom_id, interval_start, blacklist_chrom_id, blacklist_start) and smaller_equal(interval_chrom_id, interval_end, blacklist_chrom_id, blacklist_end):
            increment_interval = True

        #elif interval_start < blacklist_start and interval_end <= blacklist_end:
        elif smaller(interval_chrom_id, interval_start, blacklist_chrom_id, blacklist_start) and smaller_equal(interval_chrom_id, interval_end, blacklist_chrom_id, blacklist_end):
            chrom_id_out_view[k] = interval_chrom_id
            start_out_view[k] = interval_start
            end_out_view[k] = blacklist_start
            value_out_view[k] = interval_value
            k += 1

            interval_start = blacklist_end
            # interval_end = interval_end
            # result.append((interval_start, blacklist_start))
            # intervals[i] = (blacklist_end, interval_end)
        #
        # elif interval_start < blacklist_start and interval_end > blacklist_end:
        elif smaller(interval_chrom_id, interval_start, blacklist_chrom_id, blacklist_start) and larger(interval_chrom_id, interval_end, blacklist_chrom_id, blacklist_end):
            chrom_id_out_view[k] = interval_chrom_id
            start_out_view[k] = interval_start
            end_out_view[k] = blacklist_start
            value_out_view[k] = interval_value
            k += 1
            increment_blacklist = True
            interval_start = blacklist_end

        #elif interval_start >= blacklist_start and interval_end > blacklist_end:
        elif larger_equal(interval_chrom_id, interval_start, blacklist_chrom_id, blacklist_start) and larger(interval_chrom_id, interval_end, blacklist_chrom_id, blacklist_end):
            interval_start = blacklist_end
            #intervals[i] = (blacklist_end, interval_end)
        #elif interval_start < blacklist_start and interval_end > blacklist_end:
        elif smaller(interval_chrom_id, interval_start, blacklist_chrom_id, blacklist_start) and larger(interval_chrom_id, interval_end, blacklist_chrom_id, blacklist_end):
            chrom_id_out_view[k] = interval_chrom_id
            start_out_view[k] = interval_start
            end_out_view[k] = blacklist_start
            value_out_view[k] = interval_value
            k += 1
            #result.append((interval_start, blacklist_start))
            interval_start = blacklist_end
            increment_blacklist = True

    return chrom_id_out[:k], start_out[:k], end_out[:k], value_out[:k]

cdef bint smaller(UINT32_t chrom1, UINT32_t pos1, UINT32_t chrom2, UINT32_t pos2):
    if chrom1 < chrom2:
        return True
    elif chrom1 == chrom2:
        return pos1 < pos2
    else:
        return False

cdef bint larger(UINT32_t chrom1, UINT32_t pos1, UINT32_t chrom2, UINT32_t pos2):
    if chrom1 > chrom2:
        return True
    elif chrom1 == chrom2:
        return pos1 > pos2
    else:
        return False

cdef bint smaller_equal(UINT32_t chrom1, UINT32_t pos1, UINT32_t chrom2, UINT32_t pos2):
    return not larger(chrom1, pos1, chrom2, pos2)

cdef bint larger_equal(UINT32_t chrom1, UINT32_t pos1, UINT32_t chrom2, UINT32_t pos2):
    return not smaller(chrom1, pos1, chrom2, pos2)
