from typing import Iterable
from typing import Iterator

import numpy as np
import numpy.typing as npt

from bigwig_loader.batch import Batch


class QueryBatchGenerator:
    def __init__(
        self,
        genomic_location_sampler: Iterable[tuple[str, int, str] | tuple[str, int]],
        center_bin_to_predict: int,
        batch_size: int,
        track_sampler: Iterable[list[int]] | None = None,
    ):
        self.genomic_location_sampler = genomic_location_sampler
        self.center_bin_to_predict = center_bin_to_predict

        self.track_sampler = None
        if track_sampler is not None:
            self.track_sampler = iter(track_sampler)
        self.batch_size = batch_size

    def _sample_tracks(self) -> list[int] | None:
        if self.track_sampler is None:
            return None
        return next(self.track_sampler)

    def _sample_genomic_sequences(
        self,
    ) -> tuple[
        list[str], npt.NDArray[np.int64], npt.NDArray[np.int64], list[str] | None
    ]:
        chromosomes = []
        centers = []
        sequences = []

        for _, interval in zip(range(self.batch_size), self.genomic_location_sampler):
            chromosome, center = interval[:2]
            sequence = interval[2] if len(interval) > 2 else None  # type: ignore
            chromosomes.append(chromosome)
            centers.append(center)
            sequences.append(sequence)

        start = np.array(centers, dtype=np.int64) - np.int64(
            self.center_bin_to_predict // 2
        )
        start = start.astype(np.int64)
        end = start + np.int64(self.center_bin_to_predict)
        end = end.astype(np.int64)
        return chromosomes, start, end, sequences  # type: ignore

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        chromosomes, start, end, sequences = self._sample_genomic_sequences()
        track_indices = self._sample_tracks()
        return Batch(
            chromosomes=chromosomes,
            starts=start,
            ends=end,
            sequences=sequences,
            track_indices=track_indices,
        )
