from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt

from bigwig_loader.bigwig import BigWig
from bigwig_loader.decompressor import Decoder
from bigwig_loader.memory_bank import MemoryBank
from bigwig_loader.searchsorted import interval_searchsorted


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
    stream: Optional[cp.cuda.Stream] = None,
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
        stream=stream,
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
    stream: Optional[cp.cuda.Stream] = None,
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
        stream=stream,
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
