from random import sample
from typing import Iterator


class TrackSampler:
    def __init__(self, total_number_of_tracks: int, sample_size: int):
        self.total_number_of_tracks = total_number_of_tracks
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            yield sorted(sample(range(self.total_number_of_tracks), self.sample_size))
