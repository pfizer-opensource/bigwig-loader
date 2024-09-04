from functools import cached_property

# from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt

from bigwig_loader.bigwig import BigWig
from bigwig_loader.decompressor import Decoder
from bigwig_loader.functional import load_decode_search
from bigwig_loader.intervals_to_values import intervals_to_values
from bigwig_loader.memory_bank import MemoryBank

PreprocessedReturnType = tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
]


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
    ):
        self._bigwigs = bigwigs
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
        return MemoryBank(elastic=True)

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
        track_indices: Optional[Union[Sequence[int], npt.NDArray[np.int64]]],
        stream: Optional[cp.cuda.Stream] = None,
    ) -> PreprocessedReturnType:
        if track_indices is None:
            bigwigs = self._bigwigs
        else:
            bigwigs = [self._bigwigs[i] for i in track_indices]
        return load_decode_search(
            bigwigs=bigwigs,
            chromosomes=chromosomes,
            start=start,
            end=end,
            memory_bank=self.memory_bank,
            decoder=self.decoder,
            local_to_global=self._local_to_global,
            stream=stream,
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
