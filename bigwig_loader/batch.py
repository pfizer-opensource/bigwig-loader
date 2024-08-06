from functools import cached_property
from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt

from bigwig_loader.decompressor import Decoder
from bigwig_loader.intervals_to_values import intervals_to_values
from bigwig_loader.memory_bank import MemoryBank
from bigwig_loader.memory_bank import create_memory_bank
from bigwig_loader.searchsorted import interval_searchsorted

if TYPE_CHECKING:
    from bigwig_loader.bigwig import BigWig


class BatchProcessor:
    def __init__(
        self,
        bigwigs: Sequence["BigWig"],
        max_rows_per_chunk: int,
        local_to_global: Callable[
            [
                Union[Sequence[str], npt.NDArray[np.generic]],
                Union[Sequence[int], npt.NDArray[np.int64]],
            ],
            npt.NDArray[np.int64],
        ],
        local_chrom_ids_to_offset_matrix: cp.ndarray,
        use_cufile: bool = True,
    ):
        self._bigwigs = bigwigs
        self._use_cufile = use_cufile
        self._max_rows_per_chunk = max_rows_per_chunk
        self._local_to_global = local_to_global
        self._local_chrom_ids_to_offset_matrix = local_chrom_ids_to_offset_matrix
        self._out: cp.ndarray = cp.zeros((len(self._bigwigs), 1, 1), dtype=cp.float32)

    @cached_property
    def decoder(self) -> Decoder:
        return Decoder(
            max_rows_per_chunk=self._max_rows_per_chunk,
            max_uncompressed_chunk_size=self._max_rows_per_chunk * 12 + 24,
            chromosome_offsets=self._local_chrom_ids_to_offset_matrix,
        )

    @cached_property
    def memory_bank(self) -> MemoryBank:
        return create_memory_bank(elastic=True, use_cufile=self._use_cufile)

    def _get_out_tensor(self, batch_size: int, sequence_length: int) -> cp.ndarray:
        """Resuses a reserved tensor if possible (when out shape is constant),
        otherwise creates a new one.
        args:
            batch_size: batch size
            sequence_length: length of genomic sequence
         returns:
            tensor of shape (number of bigwig files, batch_size, sequence_length)
        """

        shape = (len(self._bigwigs), batch_size, sequence_length)
        if self._out.shape != shape:
            self._out = cp.zeros(shape, dtype=cp.float32)
        return self._out

    def preprocess(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
    ) -> tuple[
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
    ]:
        return load_decode_search(
            bigwigs=self._bigwigs,
            chromosomes=chromosomes,
            start=start,
            end=end,
            memory_bank=self.memory_bank,
            decoder=self.decoder,
            local_to_global=self._local_to_global,
        )

    def get_batch(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
        window_size: int = 1,
        scaling_factors_cupy: Optional[cp.ndarray] = None,
        out: Optional[cp.ndarray] = None,
    ) -> cp.ndarray:
        (
            start_data,
            end_data,
            value_data,
            abs_start,
            abs_end,
            found_starts,
            found_ends,
        ) = load_decode_search(
            bigwigs=self._bigwigs,
            chromosomes=chromosomes,
            start=start,
            end=end,
            memory_bank=self.memory_bank,
            decoder=self.decoder,
            local_to_global=self._local_to_global,
        )

        if out is None:
            sequence_length = (end[0] - start[0]) // window_size
            out = self._get_out_tensor(len(start), sequence_length)

        out = intervals_to_values(
            array_start=start_data,
            array_end=end_data,
            array_value=value_data,
            found_starts=found_starts,
            found_ends=found_ends,
            query_starts=abs_start,
            query_ends=abs_end,
            window_size=window_size,
            out=out,
        )
        batch = cp.transpose(out, (1, 0, 2))
        if scaling_factors_cupy is not None:
            batch *= scaling_factors_cupy

        return batch


def load(
    bigwigs: Sequence["BigWig"],
    chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
    start: Union[Sequence[int], npt.NDArray[np.int64]],
    end: Union[Sequence[int], npt.NDArray[np.int64]],
    memory_bank: MemoryBank,
    local_to_global: Callable[
        [
            Union[Sequence[str], npt.NDArray[np.generic]],
            Union[Sequence[int], npt.NDArray[np.int64]],
        ],
        npt.NDArray[np.int64],
    ],
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Load a batch of data from the bigwig files."""
    memory_bank.reset()

    abs_start = local_to_global(chromosomes, start)
    abs_end = local_to_global(chromosomes, end)

    n_chunks_per_bigwig = []
    bigwig_ids = []
    for bigwig in bigwigs:
        offsets, sizes = bigwig.get_batch_offsets_and_sizes_with_global_positions(
            abs_start, abs_end
        )
        n_chunks_per_bigwig.append(len(offsets))
        bigwig_ids.extend([bigwig.id] * len(offsets))
        # read chunks into preallocated memory
        memory_bank.add_many(
            bigwig.store.file_handle,
            offsets,
            sizes,
            skip_bytes=2,
        )

    abs_end = cp.asarray(abs_end, dtype=cp.uint32)
    abs_start = cp.asarray(abs_start, dtype=cp.uint32)

    # bring the gpu
    bigwig_ids = cp.asarray(bigwig_ids, dtype=cp.uint32)
    n_chunks_per_bigwig = cp.asarray(n_chunks_per_bigwig, dtype=cp.uint32)

    comp_chunk_pointers, compressed_chunk_sizes = memory_bank.to_gpu()

    return (
        abs_start,
        abs_end,
        bigwig_ids,
        comp_chunk_pointers,
        n_chunks_per_bigwig,
        compressed_chunk_sizes,
    )


def load_and_decode(
    bigwigs: Sequence["BigWig"],
    chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
    start: Union[Sequence[int], npt.NDArray[np.int64]],
    end: Union[Sequence[int], npt.NDArray[np.int64]],
    memory_bank: MemoryBank,
    local_to_global: Callable[
        [
            Union[Sequence[str], npt.NDArray[np.generic]],
            Union[Sequence[int], npt.NDArray[np.int64]],
        ],
        npt.NDArray[np.int64],
    ],
    decoder: Decoder,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    (
        abs_start,
        abs_end,
        bigwig_ids,
        comp_chunk_pointers,
        n_chunks_per_bigwig,
        compressed_chunk_sizes,
    ) = load(
        bigwigs=bigwigs,
        chromosomes=chromosomes,
        start=start,
        end=end,
        memory_bank=memory_bank,
        local_to_global=local_to_global,
    )

    start_data, end_data, value_data, bigwig_start_indices = decoder.decode_batch(
        bigwig_ids=bigwig_ids,
        comp_chunk_pointers=comp_chunk_pointers,
        compressed_chunk_sizes=compressed_chunk_sizes,
        n_chunks_per_bigwig=n_chunks_per_bigwig,
    )

    return (
        abs_start,
        abs_end,
        start_data,
        end_data,
        value_data,
        bigwig_start_indices,
    )


def load_decode_search(
    bigwigs: Sequence["BigWig"],
    chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
    start: Union[Sequence[int], npt.NDArray[np.int64]],
    end: Union[Sequence[int], npt.NDArray[np.int64]],
    memory_bank: MemoryBank,
    local_to_global: Callable[
        [
            Union[Sequence[str], npt.NDArray[np.generic]],
            Union[Sequence[int], npt.NDArray[np.int64]],
        ],
        npt.NDArray[np.int64],
    ],
    decoder: Decoder,
) -> tuple[
    cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray
]:
    (
        abs_start,
        abs_end,
        start_data,
        end_data,
        value_data,
        bigwig_start_indices,
    ) = load_and_decode(
        bigwigs=bigwigs,
        chromosomes=chromosomes,
        start=start,
        end=end,
        memory_bank=memory_bank,
        local_to_global=local_to_global,
        decoder=decoder,
    )

    found_starts, found_ends = interval_searchsorted(
        array_start=start_data,
        array_end=end_data,
        query_starts=abs_start,
        query_ends=abs_end,
        start_indices=bigwig_start_indices,
        absolute_indices=True,
    )

    return (
        start_data,
        end_data,
        value_data,
        abs_start,
        abs_end,
        found_starts,
        found_ends,
    )
