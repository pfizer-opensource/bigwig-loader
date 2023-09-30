import numpy as np
import pandas as pd
from natsort import natsorted

cimport cython
cimport numpy as cnp

ctypedef cnp.npy_uint32 UINT32_t




cdef list ID_TO_KEY_CANONICAL = [f"chr{x}" for x in range(1, 23)] + ["chrX", "chrY", "chrM"]
cdef set CANONICAL_CHROM_KEYS = set(ID_TO_KEY_CANONICAL)

def merge_interval_dataframe(intervals: pd.DataFrame, bint is_sorted=False, UINT32_t allow_gap=0):
    chrom, start, end, value = merge_intervals_with_chrom_keys(intervals["chrom"].values,
                                                               intervals["start"].values,
                                                               intervals["end"].values,
                                                               intervals["value"].values,
                                                               is_sorted,
                                                               allow_gap)
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})

def merge_intervals_with_chrom_keys(cnp.ndarray[object, ndim=1] chroms,
                                    cnp.ndarray[UINT32_t, ndim=1] starts,
                                    cnp.ndarray[UINT32_t, ndim=1] ends,
                                    cnp.ndarray[cnp.float32_t, ndim=1] values,
                                    bint is_sorted,
                                    UINT32_t allow_gap=0):
    id_to_key = np.array(ID_TO_KEY_CANONICAL + natsorted(set(chroms) - CANONICAL_CHROM_KEYS), dtype=object)
    key_to_id = {key: i for i, key in enumerate(id_to_key)}
    chrom_ids = np.array([key_to_id[key] for key in chroms], dtype="uint32")
    chrom_ids_out, starts_out, ends_out, values_out = merge_intervals(chrom_ids, starts, ends, values, is_sorted, allow_gap)
    chroms_out = id_to_key[chrom_ids_out]
    return chroms_out, starts_out, ends_out, values_out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef merge_intervals(cnp.ndarray[cnp.npy_uint32, ndim = 1] chrom_ids,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] starts,
                      cnp.ndarray[cnp.npy_uint32, ndim = 1] ends,
                      cnp.ndarray[cnp.float32_t, ndim = 1] values,
                      bint is_sorted,
                      UINT32_t allow_gap):
    cdef:
        size_t i
        size_t j
        size_t n_intervals
        UINT32_t chrom_id
        UINT32_t start
        UINT32_t end
        float value
        UINT32_t chrom_id_next
        UINT32_t start_next
        UINT32_t end_next
        UINT32_t start_new
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

    if not is_sorted:
        idx = np.lexsort([starts, chrom_ids])
        chrom_ids = chrom_ids[idx]
        starts = starts[idx]
        ends = ends[idx]
        values = values[idx]

    i = 1
    j = 0

    cdef UINT32_t[:] chrom_id_view = chrom_ids
    cdef UINT32_t[:] start_view = starts
    cdef UINT32_t[:] end_view = ends
    cdef float[:] value_view = values

    start_new = starts[0]
    value_new = zero
    while i < n_intervals:
        chrom_id = chrom_id_view[i - 1]
        start = start_view[i - 1]
        end = end_view[i - 1]
        value = value_view[i - 1]

        chrom_id_next = chrom_id_view[i]
        start_next = start_view[i]
        end_next = end_view[i]

        value_new = max(value, value_new)

        # i.e. there is a gap
        if start_next > end + allow_gap or chrom_id != chrom_id_next:
            chrom_id_out_view[j] = chrom_id
            start_out_view[j] = start_new
            end_out_view[j] = end
            value_out_view[j] = value_new
            start_new = start_next
            value_new = zero
            j += 1
        i += 1

    chrom_id_out_view[j] = chrom_id
    start_out_view[j] = start_new
    end_out_view[j] = end_next
    value_out_view[j] = value_new
    j += 1

    return chrom_id_out[:j], start_out[:j], end_out[:j], value_out[:j]
