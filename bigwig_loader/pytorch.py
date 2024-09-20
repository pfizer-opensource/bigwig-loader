from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from bigwig_loader.batch import Batch
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.dataset import BigWigDataset


class PytorchBatch:
    """Batch
    This is a simple container object to hold on to a set of arrays
    that represent a batch of data. It serves as the query to bigwig_loader,
    therefore the minimum init args are chromosomes, starts and ends, which
    is really everything bigwig_loader really needs.

    It is used to pass data through the input and output queues of the
    dataloader that works with threads and cuda streams. At some point it
    gets cumbersome to keep track of the order of the arrays, so this
    object is used to make that a bit simpler.

    Args:
        chromosomes: 1D array of size batch_size ["chr1", "chr2", ...],
        starts: 1D array of size batch_size [0, 42, ...],
        ends: 1D array of size batch_size [100, 142, ...],
        track_indices (Optional): which tracks to include in this batch
            (index should correspond to the order in
            bigwig_loader.collection.BigWigCollection). When None, all
            tracks are included.
        sequences (Optional): ["ACTAGANTG", "CCTTGAGT", ...].
        values (cp.ndarray | None): The values of the batch: the output matrix
        bigwig_loader produces. size: (batch_size, n_tracks, n_values)
        other_batched (list of Arrays): other arrays that share
            the batch_dimension with chromosomes, starts, ends, sequences and
            values. Here for convenience. When creating a slice of Batch,
            these arrays are sliced in the same way the previously mentioned
            arrays are sliced.
        other (Any): Any other data to hold on to for the batch. Can be anything
            No slicing is performed on this object when the Batch is sliced.
    """

    def __init__(
        self,
        chromosomes: Any,
        starts: Any,
        ends: Any,
        values: torch.Tensor,
        track_indices: torch.Tensor | None,
        sequences: torch.Tensor | list[str] | None,
        other_batched: Any | None,
        other: Any,
        track_names: Sequence[str | Path] | None = None,
    ):
        self.chromosomes = chromosomes
        self.starts = starts
        self.ends = ends
        self.sequences = sequences
        self.track_indices = track_indices
        self.values = values
        self.other_batched = other_batched
        self.other = other
        self.track_names = track_names

    @classmethod
    def from_batch(cls, batch: Batch) -> "PytorchBatch":
        if batch.other_batched is not None:
            other_batched = (
                [cls._convert_if_possible(tensor) for tensor in batch.other_batched],
            )
        else:
            other_batched = None
        return PytorchBatch(
            chromosomes=cls._convert_if_possible(batch.chromosomes),
            starts=cls._convert_if_possible(batch.starts),
            ends=cls._convert_if_possible(batch.ends),
            values=cls._convert_if_possible(batch.values),
            track_indices=cls._convert_if_possible(batch.track_indices),
            sequences=cls._convert_if_possible(batch.sequences),
            other_batched=other_batched,
            other=cls._convert_if_possible(batch.other),
            track_names=batch.track_names,
        )

    @staticmethod
    def _convert_if_possible(tensor: Any) -> Any:
        if isinstance(tensor, cp.ndarray) or isinstance(tensor, np.ndarray):
            return torch.as_tensor(tensor)
        return tensor


GENOMIC_SEQUENCE_TYPE = Union[torch.Tensor, list[str], None]
BATCH_TYPE = Union[
    tuple[GENOMIC_SEQUENCE_TYPE, torch.Tensor],
    tuple[
        GENOMIC_SEQUENCE_TYPE, torch.Tensor, torch.Tensor, Sequence[str | Path] | None
    ],
    PytorchBatch,
]


class PytorchBigWigDataset(IterableDataset[BATCH_TYPE]):

    """
    Pytorch IterableDataset over FASTA files and BigWig profiles.
    Args:
        regions_of_interest: pandas dataframe with intervals within the chromosomes to
            sample from. These intervals are not intervals of sequence_length
            but regions of the chromosome that were selected up front. For
            instance regions in which called peaks are located or a threshold
            value is exceeded.
        collection: instance of bigwig_loader.BigWigCollection or path to BigWig
            Directory or list of BigWig files.
        reference_genome_path: path to fasta file containing the reference genome.
        sequence_length: number of base pairs in input sequence
        center_bin_to_predict: if given, only do prediction on a central window. Should be
            smaller than or equal to sequence_length. If not given will be the same as
            sequence_length.
        window_size: used to down sample the resolution of the target from sequence_length
        moving_average_window_size: window size for moving average on the target. Can
            help too smooth out the target. Default: 1, which means no smoothing. If
            used in combination with window_size, the target is first downsampled and
            then smoothed.
        batch_size: batch size
        super_batch_size: batch size that is used in the background to load data from
            bigwig files. Should be larger than batch_size. If None, it will be equal to
            batch_size.
        batches_per_epoch: because the length of an epoch is slightly arbitrary here,
            the number of batches can be set by hand. If not the number of batches per
            epoch will be (totol number of bases in combined intervals) // sequence_length // batch_size
        maximum_unknown_bases_fraction: maximum number of bases in an input sequence that
            is unknown.
        sequence_encoder: encoder to apply to the sequence. Default: bigwig_loader.util.onehot_sequences
        position_samples_buffer_size: number of intervals picked up front by the position sampler.
            When all intervals are used, new intervals are picked.
        sub_sample_tracks: int, if set a  different random set of tracks is selected in each
            superbatch from the total number of tracks. The indices corresponding to those tracks
            are returned in the output.
        n_threads: number of python threads / cuda streams to use for loading the data to
            GPU. More threads means that more IO can take place while the GPU is busy doing
            calculations (decompressing or neural network training for example). More threads
            also means a higher GPU memory usage. Default: 4
        return_batch_objects: if True, the batches will be returned as instances of
            bigwig_loader.pytorch.PytorchBatch
    """

    def __init__(
        self,
        regions_of_interest: pd.DataFrame,
        collection: Union[str, Sequence[str], Path, Sequence[Path], BigWigCollection],
        reference_genome_path: Path,
        sequence_length: int = 1000,
        center_bin_to_predict: Optional[int] = 200,
        window_size: int = 1,
        moving_average_window_size: int = 1,
        batch_size: int = 256,
        super_batch_size: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        maximum_unknown_bases_fraction: float = 0.1,
        sequence_encoder: Optional[
            Union[Callable[[Sequence[str]], Any], Literal["onehot"]]
        ] = "onehot",
        file_extensions: Sequence[str] = (".bigWig", ".bw"),
        crawl: bool = True,
        first_n_files: Optional[int] = None,
        position_sampler_buffer_size: int = 100000,
        repeat_same_positions: bool = False,
        sub_sample_tracks: Optional[int] = None,
        n_threads: int = 4,
        return_batch_objects: bool = False,
    ):
        super().__init__()
        self._dataset = BigWigDataset(
            regions_of_interest=regions_of_interest,
            collection=collection,
            reference_genome_path=reference_genome_path,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            window_size=window_size,
            moving_average_window_size=moving_average_window_size,
            batch_size=batch_size,
            super_batch_size=super_batch_size,
            batches_per_epoch=batches_per_epoch,
            maximum_unknown_bases_fraction=maximum_unknown_bases_fraction,
            sequence_encoder=sequence_encoder,
            file_extensions=file_extensions,
            crawl=crawl,
            first_n_files=first_n_files,
            position_sampler_buffer_size=position_sampler_buffer_size,
            repeat_same_positions=repeat_same_positions,
            sub_sample_tracks=sub_sample_tracks,
            n_threads=n_threads,
            return_batch_objects=True,
        )
        self._return_batch_objects = return_batch_objects

    def __iter__(
        self,
    ) -> Iterator[BATCH_TYPE]:
        for batch in self._dataset:
            pytorch_batch = PytorchBatch.from_batch(batch)  # type: ignore
            if self._return_batch_objects:
                yield pytorch_batch
            elif pytorch_batch.track_indices is None:
                yield pytorch_batch.sequences, pytorch_batch.values
            else:
                yield pytorch_batch.sequences, pytorch_batch.values, pytorch_batch.track_indices, pytorch_batch.track_names

    def reset_gpu(self) -> None:
        self._dataset.reset_gpu()
