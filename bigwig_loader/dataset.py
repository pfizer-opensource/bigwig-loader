import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import pandas as pd

from bigwig_loader.batch import Batch
from bigwig_loader.batch import IntSequenceType
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.cupy_functions import moving_average
from bigwig_loader.input_generator import QueryBatchGenerator
from bigwig_loader.sampler.genome_sampler import GenomicSequenceSampler
from bigwig_loader.sampler.position_sampler import RandomPositionSampler
from bigwig_loader.sampler.track_sampler import TrackSampler
from bigwig_loader.streamed_dataset import StreamedDataloader
from bigwig_loader.util import onehot_sequences_cupy

ENCODERS: dict[str, Callable[[Sequence[str]], Any]] = {"onehot": onehot_sequences_cupy}


class BigWigDataset:
    """
    Dataset over FASTA files and BigWig profiles.
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
            bigwig files. Should be larget than or equal to batch_size. If None, it will
            be equal to batch_size.
        batches_per_epoch: because the length of an epoch is slightly arbitrary here,
            the number of batches can be set by hand. If not the number of batches per
            epoch will be (totol number of bases in combined intervals) // sequence_length // batch_size
        maximum_unknown_bases_fraction: maximum number of bases in an input sequence that
            is unknown.
        sequence_encoder: encoder to apply to the sequence. Default: bigwig_loader.util.onehot_sequences
        file_extensions: load files with these extensions (default .bw and .bigWig)
        crawl: whether to search in sub-directories for BigWig files
        scale: Optional, dictionary with scaling factors for each BigWig file.
            If None, no scaling is done. Keys can be (partial) file paths. See
            bigwig_loader.path.match_key_to_path for more information about how
            dict keys are mapped to paths.
        first_n_files: Only use the first n files (handy for debugging on less tasks)
        position_sampler_buffer_size: number of intervals picked up front by the position sampler.
            When all intervals are used, new intervals are picked.
        repeat_same_positions: if True the positions sampler does not draw a new random collection
            of positions when the buffer runs out, but repeats the same samples. Can be used to
            check whether network can overfit.
        sub_sample_tracks: int, if set a  different random set of tracks is selected in each
            superbatch from the total number of tracks. The indices corresponding to those tracks
            are returned in the output.
        n_threads: number of python threads / cuda streams to use for loading the data to
            GPU. More threads means that more IO can take place while the GPU is busy doing
            calculations (decompressing or neural network training for example). More threads
            also means a higher GPU memory usage.
        return_batch_objects: if True, the batches will be returned as instances of
            bigwig_loader.batch.Batch
    """

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
        n_threads: int = 4,
        return_batch_objects: bool = False,
    ):
        super().__init__()

        self.regions_of_interest = regions_of_interest
        if isinstance(collection, BigWigCollection):
            self._bigwig_collection: Optional[BigWigCollection] = collection
            self._bigwig_path: Optional[
                Union[str, Sequence[str], Path, Sequence[Path]]
            ] = None
        else:
            self._bigwig_collection = None
            self._bigwig_path = collection
        self.reference_genome_path = reference_genome_path
        self.sequence_length = sequence_length
        if center_bin_to_predict:
            self.center_bin_to_predict = center_bin_to_predict
        else:
            self.center_bin_to_predict = sequence_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.super_batch_size = super_batch_size or batch_size
        self.batches_per_epoch = (
            batches_per_epoch
            or (regions_of_interest["end"] - regions_of_interest["start"]).sum()
            // sequence_length
            // batch_size
        )
        self._n = 0
        self.maximum_unknown_bases_fraction = maximum_unknown_bases_fraction
        if sequence_encoder in ENCODERS:
            self._sequence_encoder = ENCODERS[sequence_encoder]  # type: ignore
        else:
            self._sequence_encoder = sequence_encoder  # type: ignore
        self._first_n_files = first_n_files
        self._file_extensions = file_extensions
        self._crawl = crawl
        self._scale = scale
        self._position_sampler_buffer_size = position_sampler_buffer_size
        self._repeat_same_positions = repeat_same_positions
        self._moving_average_window_size = moving_average_window_size
        self._sub_sample_tracks = sub_sample_tracks
        self._n_threads = n_threads
        self._return_batch_objects = return_batch_objects

    def _create_dataloader(self) -> StreamedDataloader:
        position_sampler = RandomPositionSampler(
            regions_of_interest=self.regions_of_interest,
            buffer_size=self._position_sampler_buffer_size,
            repeat_same=self._repeat_same_positions,
        )

        sequence_sampler = GenomicSequenceSampler(
            reference_genome_path=self.reference_genome_path,
            sequence_length=self.sequence_length,
            position_sampler=position_sampler,
            maximum_unknown_bases_fraction=self.maximum_unknown_bases_fraction,
        )
        track_sampler = None
        if self._sub_sample_tracks is not None:
            track_sampler = TrackSampler(
                total_number_of_tracks=len(self.bigwig_collection),
                sample_size=self._sub_sample_tracks,
            )

        query_batch_generator = QueryBatchGenerator(
            genomic_location_sampler=sequence_sampler,
            center_bin_to_predict=self.center_bin_to_predict,
            batch_size=self.super_batch_size,
            track_sampler=track_sampler,
        )

        return StreamedDataloader(
            input_generator=query_batch_generator,
            collection=self.bigwig_collection,
            num_threads=self._n_threads,
            queue_size=self._n_threads + 1,
            slice_size=self.batch_size,
            window_size=self.window_size,
        )

    def __iter__(
        self,
    ) -> Iterator[
        tuple[cp.ndarray | list[str] | None, cp.ndarray]
        | tuple[
            cp.ndarray | list[str] | None,
            cp.ndarray,
            IntSequenceType,
            Sequence[str | Path] | None,
        ]
        | Batch
    ]:
        with self._create_dataloader() as dataloader:
            for i, batch in enumerate(dataloader):
                values = moving_average(batch.values, self._moving_average_window_size)
                if batch.sequences is not None and self._sequence_encoder is not None:
                    sequences = self._sequence_encoder(batch.sequences)
                else:
                    sequences = batch.sequences
                if batch.track_indices is not None:
                    track_names = [
                        self.bigwig_collection.bigwig_paths[i]
                        for i in batch.track_indices
                    ]
                else:
                    track_names = self.bigwig_collection.bigwig_paths
                if self._return_batch_objects:
                    batch.sequences = sequences
                    batch.values = values
                    batch.track_names = track_names
                    yield batch
                elif batch.track_indices is not None:
                    yield sequences, values, batch.track_indices, track_names
                else:
                    yield sequences, values
                if i == self.batches_per_epoch - 1:
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

    def reset_gpu(self) -> None:
        warnings.warn(
            "reset_gpu is deprecated because it is not necessay anymore. A new loader"
            "instance is create each time __iter__ is called,",
            DeprecationWarning,
        )
