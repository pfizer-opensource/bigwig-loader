from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numcodecs
import numpy as np
import numpy.typing as npt
from numcodecs.compat import ensure_contiguous_ndarray_like

codec = numcodecs.registry.get_codec(
    {
        "id": "nvcomp_batch",
        "algorithm": "Deflate",
    }
)


class Decoder:
    # based on: `kvikio.nvcomp_codec.NvCompBatchCodec`

    def __init__(
        self,
        max_num_chunks: int,
        max_rows_per_chunk: int,
        max_uncompressed_chunk_size: int = (2048 * 12),
        chromosome_offsets: Optional[
            Union[Sequence[int], npt.NDArray[np.int64]]
        ] = None,
    ):
        self.max_uncompressed_chunk_size = max_uncompressed_chunk_size

        self.uncomp_chunks = [
            cp.empty(max_uncompressed_chunk_size, dtype=cp.uint8)
            for _ in range(max_num_chunks)
        ]
        self.uncomp_chunk_ptrs = cp.array(
            [c.data.ptr for c in self.uncomp_chunks], dtype=cp.uint64
        )
        self.uncomp_chunk_sizes = cp.array(
            [max_uncompressed_chunk_size] * max_num_chunks, dtype=cp.uint64
        )
        self.actual_uncomp_chunk_sizes = cp.empty(max_num_chunks, dtype=cp.uint64)
        self.statuses = cp.empty(max_num_chunks, dtype=cp.int32)
        self.max_rows_per_chunk = max_rows_per_chunk

        self.range_of_indexes = cp.arange(max_rows_per_chunk)
        if chromosome_offsets is not None:
            self.chromosome_offsets = cp.asarray(chromosome_offsets, dtype="uint32")
        else:
            self.chromosome_offsets = None

    def decode(
        self,
        gpu_array: cp.ndarray,
        comp_chunks: cp.ndarray,
        comp_chunk_sizes: cp.ndarray,
        bigwig_ids: Optional[cp.ndarray] = None,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        num_chunks = len(comp_chunks)

        max_chunk_size = int(self.max_uncompressed_chunk_size)

        # Get temp buffer size.
        temp_size = codec._algo.get_decompress_temp_size(num_chunks, max_chunk_size)

        temp_buf = cp.empty(temp_size, dtype=cp.uint8)

        uncomp_chunks = self.uncomp_chunks[:num_chunks]
        uncomp_chunk_ptrs = self.uncomp_chunk_ptrs[:num_chunks]
        uncomp_chunk_sizes = self.uncomp_chunk_sizes[:num_chunks]

        actual_uncomp_chunk_sizes = self.actual_uncomp_chunk_sizes[:num_chunks]

        statuses = self.statuses[:num_chunks]

        codec._algo.decompress(
            comp_chunks,  # const void* const*
            comp_chunk_sizes,  # const size_t*
            num_chunks,  # ??
            temp_buf,  #
            uncomp_chunk_ptrs,
            uncomp_chunk_sizes,
            actual_uncomp_chunk_sizes,
            statuses,
            codec._stream,
        )

        chrom_ids, start, end, value, n_rows_for_chunks = self.post_process(
            cp.concatenate(uncomp_chunks), bigwig_ids
        )
        return chrom_ids, start, end, value, n_rows_for_chunks

    def post_process(
        self, arr: cp.ndarray, bigwig_ids: Optional[cp.ndarray] = None
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        max_rows_per_chunk = self.max_rows_per_chunk
        n_rows_for_chunks = arr.view(dtype="u2")[
            11 :: (max_rows_per_chunk * 12 + 24) // 2
        ]

        chrom_ids = arr.view(dtype="<u4")[0 :: (max_rows_per_chunk * 12 + 24) // 4]

        start = (
            arr.view(dtype="<u4")
            .reshape((-1, 3))[:, 0]
            .reshape((-1, max_rows_per_chunk + 2))[:, 2:]
        )
        end = (
            arr.view(dtype="<u4")
            .reshape((-1, 3))[:, 1]
            .reshape((-1, max_rows_per_chunk + 2))[:, 2:]
        )
        value = (
            arr.view(dtype="<f4")
            .reshape((-1, 3))[:, 2]
            .reshape((-1, max_rows_per_chunk + 2))[:, 2:]
        )

        if self.chromosome_offsets is not None and bigwig_ids is not None:
            chrom_offsets = self.chromosome_offsets[bigwig_ids, chrom_ids]
            chrom_offsets = chrom_offsets[..., cp.newaxis]
            start += chrom_offsets
            end += chrom_offsets

        if (n_rows_for_chunks == max_rows_per_chunk).all():
            start = cp.ravel(start)
            end = cp.ravel(end)
            value = cp.ravel(value)
        else:
            mask = cp.greater.outer(n_rows_for_chunks, self.range_of_indexes)
            start = start[mask]
            end = end[mask]
            value = value[mask]

        return chrom_ids, start, end, value, n_rows_for_chunks
