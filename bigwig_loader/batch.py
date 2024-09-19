from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt

StrSequenceType = Sequence[str]
IntSequenceType = Union[Sequence[int], npt.NDArray[np.int64]]
IntervalType = tuple[StrSequenceType, IntSequenceType, IntSequenceType]


class Batch:
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
        chromosomes: StrSequenceType,
        starts: IntSequenceType,
        ends: IntSequenceType,
        track_indices: IntSequenceType | None = None,
        sequences: StrSequenceType | None = None,
        values: cp.ndarray | None = None,
        track_names: Sequence[str | Path] | None = None,
        other_batched: Sequence[Sequence[Any] | npt.NDArray[np.generic] | cp.ndarray]
        | None = None,
        other: Any = None,
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
    def from_args(cls, args: Union["Batch", IntervalType, dict[str, Any]]) -> "Batch":
        if isinstance(args, cls):
            return args
        if isinstance(args, dict):
            return cls(**args)
        elif isinstance(args, tuple) or isinstance(args, list):
            if len(args) != 3:
                raise ValueError(
                    f"You are feeding the dataloader with {len(args)} elements. Expected exaxctly"
                    "3 elements (corresponding to chromosome, start, end). When more arguments should"
                    f"be passed please use bigwig_loader.batch.Batch directly."
                )
            return Batch(*args)
        raise ValueError(f"Can't create a Batch from {args}")

    def __getitem__(self, item: slice) -> "Batch":
        return Batch(
            chromosomes=self.chromosomes[item],
            starts=self.starts[item],
            ends=self.ends[item],
            track_indices=self.track_indices,
            sequences=self.sequences[item] if self.sequences is not None else None,
            values=self.values[item] if self.values is not None else None,
            track_names=self.track_names,
            other_batched=[x[item] for x in self.other_batched]
            if self.other_batched is not None
            else None,
            other=self.other,
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __repr__(self) -> str:
        n_chromosomes = len(self.chromosomes) if self.chromosomes is not None else 0
        n_starts = len(self.starts) if self.starts is not None else 0
        n_ends = len(self.ends) if self.ends is not None else 0
        n_sequences = len(self.sequences) if self.sequences is not None else 0
        value_shape = self.values.shape if self.values is not None else 0
        n_track_indices = (
            len(self.track_indices) if self.track_indices is not None else 0
        )
        return (
            f"Batch(chromosomes={n_chromosomes}, starts={n_starts}, ends={n_ends}, "
            f"sequences={n_sequences}, values={value_shape}, track_indices={n_track_indices})"
        )
