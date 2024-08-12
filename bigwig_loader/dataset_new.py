from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import pandas as pd

from bigwig_loader.batch import IntSequenceType
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.input_generator import QueryBatchGenerator
from bigwig_loader.sampler.genome_sampler import GenomicSequenceSampler
from bigwig_loader.sampler.position_sampler import RandomPositionSampler
from bigwig_loader.sampler.track_sampler import TrackSampler
from bigwig_loader.streamed_dataset import StreamedDataloader
from bigwig_loader.util import onehot_sequences_cupy

ENCODERS: dict[str, Callable[[Sequence[str]], Any]] = {"onehot": onehot_sequences_cupy}


class NewDataset:
    def __init__(
        self,
        regions_of_interest: pd.DataFrame,
        collection: Union[str, Sequence[str], Path, Sequence[Path], BigWigCollection],
        reference_genome_path: Path,
        sequence_length: int = 1000,
        center_bin_to_predict: Optional[int] = None,
        window_size: int = 1,
        moving_average_window_size: int = 1,
        batch_size: int = 256,
        super_batch_size: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        maximum_unknown_bases_fraction: float = 0.1,
        sequence_encoder: Optional[
            Union[Callable[[Sequence[str]], Any], str]
        ] = "onehot",
        file_extensions: Sequence[str] = (".bigWig", ".bw"),
        crawl: bool = True,
        scale: Optional[dict[Union[str | Path], Any]] = None,
        first_n_files: Optional[int] = None,
        position_sampler_buffer_size: int = 100000,
        repeat_same_positions: bool = False,
        sub_sample_tracks: Optional[int] = None,
    ):
        super_batch_size = super_batch_size or batch_size
        self.regions_of_interest = regions_of_interest
        if isinstance(collection, BigWigCollection):
            self._bigwig_collection: Optional[BigWigCollection] = collection
            self._bigwig_path: Optional[
                Union[str, Sequence[str], Path, Sequence[Path]]
            ] = None
        else:
            self._bigwig_collection = None
            self._bigwig_path = collection
        self.batches_per_epoch = (
            batches_per_epoch
            or (regions_of_interest["end"] - regions_of_interest["start"]).sum()
            // sequence_length
            // batch_size
        )
        self._file_extensions = file_extensions
        self._crawl = crawl
        self._scale = scale
        self._first_n_files = first_n_files
        if sequence_encoder in ENCODERS:
            self._sequence_encoder = ENCODERS[sequence_encoder]  # type: ignore
        else:
            self._sequence_encoder = sequence_encoder  # type: ignore

        position_sampler = RandomPositionSampler(
            regions_of_interest=regions_of_interest,
            buffer_size=position_sampler_buffer_size,
            repeat_same=repeat_same_positions,
        )

        sequence_sampler = GenomicSequenceSampler(
            reference_genome_path=reference_genome_path,
            sequence_length=sequence_length,
            position_sampler=position_sampler,
            maximum_unknown_bases_fraction=maximum_unknown_bases_fraction,
        )
        self.sub_sample_tracks = sub_sample_tracks
        track_sampler = None
        if sub_sample_tracks is not None:
            track_sampler = TrackSampler(
                total_number_of_tracks=11, sample_size=len(self.bigwig_collection)
            )

        if center_bin_to_predict is None:
            center_bin_to_predict = sequence_length

        query_batch_generator = QueryBatchGenerator(
            genomic_location_sampler=sequence_sampler,
            center_bin_to_predict=center_bin_to_predict,
            batch_size=super_batch_size,
            track_sampler=track_sampler,
        )

        self.dataloader = StreamedDataloader(
            input_generator=query_batch_generator,
            collection=self.bigwig_collection,
            num_threads=4,
            queue_size=4,
            slice_size=batch_size,
            window_size=window_size,
        )

    def __iter__(
        self,
    ) -> Iterator[
        tuple[cp.ndarray | list[str] | None, cp.ndarray]
        | tuple[cp.ndarray | list[str] | None, cp.ndarray, IntSequenceType]
    ]:
        for i, batch in enumerate(self.dataloader):
            if batch.sequences is not None and self._sequence_encoder is not None:
                sequences = self._sequence_encoder(batch.sequences)
            else:
                sequences = batch.sequences
            if batch.track_indices is not None:
                yield sequences, batch.values, batch.track_indices
            else:
                yield sequences, batch.values
            if i == self.batches_per_epoch:
                self.dataloader.stop()
                break

    @property
    def bigwig_collection(self) -> BigWigCollection:
        """
        Setup the BigWigCollection to get values from all bigwig
        files from
        """

        if self._bigwig_collection is not None:
            return self._bigwig_collection

        elif self._bigwig_path is not None:
            self._bigwig_collection = BigWigCollection(
                self._bigwig_path,
                file_extensions=self._file_extensions,
                crawl=self._crawl,
                scale=self._scale,
                first_n_files=self._first_n_files,
            )
            return self._bigwig_collection
        else:
            raise RuntimeError(
                f"{self}._bigwig_collection and {self}._bigwig_path are bot None. At least one should be set."
            )
