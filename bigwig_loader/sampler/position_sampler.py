from typing import Iterator

import numpy as np
import pandas as pd

from bigwig_loader.util import make_cumulative_index_intervals


class RandomPositionSampler:
    def __init__(
        self,
        regions_of_interest: pd.DataFrame,
        buffer_size: int = 100000,
        repeat_same: bool = False,
    ) -> None:
        self.regions_of_interest = make_cumulative_index_intervals(regions_of_interest)
        self.buffer_size = buffer_size
        self._max_index = self.regions_of_interest.index[-1]
        self._chromosomes: list[str] = []
        self._centers: list[int] = []
        self._index = 0
        self._repeat_same = repeat_same

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return self

    def __next__(self) -> tuple[str, int]:
        if self._index >= self.buffer_size:
            self._index = 0
        if self._index == 0:
            if not self._repeat_same or not self._centers:
                self._refresh_buffer()
        chromosome = self._chromosomes[self._index]
        center = self._centers[self._index]
        self._index += 1
        return chromosome, center

    def _refresh_buffer(self) -> None:
        batch_rand = np.random.randint(
            low=0, high=self._max_index, size=self.buffer_size
        )
        containing_intervals = self.regions_of_interest.iloc[
            self.regions_of_interest.index.searchsorted(batch_rand) - 1  # type: ignore
        ]
        self._centers = list(
            containing_intervals["start"] + (batch_rand - containing_intervals.index)
        )
        self._chromosomes = list(containing_intervals["chrom"])
