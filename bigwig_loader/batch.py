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
    from bigwig_loader.collection import BigWigCollection


class Batch:
    def __init__(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
        mask: Optional[Union[Sequence[bool], npt.NDArray[np.bool_]]] = None,
    ):
        self.chromosomes = chromosomes
        self.start = start
        self.end = end
        self.mask = mask

        # needed for io
        self._memory_bank = None

        # needed for decompress
        self._comp_chunk_pointers = None
        self._compressed_chunk_sizes = None
        self._bigwig_ids = None

        # needed for searchsorted
        self._array_start = None
        self._array_end = None
        self._query_starts = None
        self._query_ends = None
        self._sizes = None

        # needed for intervals_to_values
        # self._array_start = None
        # self._array_end = None
        self._array_value = None
        self._found_starts = None
        self._found_ends = None
        # self._query_starts = None
        # self._query_ends = None


class BatchProcessor:
    _count = 0

    def __init__(self, collection: "BigWigCollection", use_cufile: bool = True):
        BatchProcessor._count += 1
        print(f"BatchProcessor created {BatchProcessor._count} times")
        self.collection = collection
        self._use_cufile = use_cufile

    @cached_property
    def decoder(self) -> Decoder:
        return Decoder(
            max_rows_per_chunk=self.collection.max_rows_per_chunk,
            max_uncompressed_chunk_size=self.collection.max_rows_per_chunk * 12 + 24,
            chromosome_offsets=self.collection.local_chrom_ids_to_offset_matrix,
        )

    @cached_property
    def memory_bank(self) -> MemoryBank:
        return create_memory_bank(elastic=True, use_cufile=self._use_cufile)

    def load(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray,]:
        return load_and_decode(
            bigwigs=self.collection.bigwigs,
            chromosomes=chromosomes,
            start=start,
            end=end,
            memory_bank=self.memory_bank,
            decoder=self.decoder,
            local_to_global=self.collection.make_positions_global,
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
            abs_start,
            abs_end,
            start_data,
            end_data,
            value_data,
            bigwig_start_indices,
        ) = self.load(chromosomes=chromosomes, start=start, end=end)

        sizes = bigwig_start_indices[1:] - bigwig_start_indices[:-1]
        sizes = sizes.astype(cp.uint32)

        found_starts, found_ends = interval_searchsorted(
            array_start=start_data,
            array_end=end_data,
            query_starts=abs_start,
            query_ends=abs_end,
            sizes=sizes,
            absolute_indices=True,
        )

        # if out is None:
        #     sequence_length = (end[0] - start[0]) // window_size
        #     out = self._get_out_tensor(len(start), sequence_length)

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


def decode(
    decoder: Decoder,
    bigwig_ids: cp.ndarray,
    comp_chunk_pointers: cp.ndarray,
    compressed_chunk_sizes: cp.ndarray,
    n_chunks_per_bigwig: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    _, start_data, end_data, value_data, n_rows_for_chunks = decoder.decode(
        comp_chunk_pointers, compressed_chunk_sizes, bigwig_ids=bigwig_ids
    )

    bigwig_starts = cp.pad(cp.cumsum(n_chunks_per_bigwig), (1, 0))
    chunk_starts = cp.pad(cp.cumsum(n_rows_for_chunks), (1, 0))
    bigwig_start_indices = chunk_starts[bigwig_starts]

    return (
        start_data,
        end_data,
        value_data,
        bigwig_start_indices,
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

    start_data, end_data, value_data, bigwig_start_indices = decode(
        decoder,
        bigwig_ids,
        comp_chunk_pointers,
        compressed_chunk_sizes,
        n_chunks_per_bigwig,
    )

    return (
        abs_start,
        abs_end,
        start_data,
        end_data,
        value_data,
        bigwig_start_indices,
    )
