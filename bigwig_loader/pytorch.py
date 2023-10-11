from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from bigwig_loader.collection import BigWigCollection
from bigwig_loader.dataset import BigWigDataset


class PytorchBigWigDataset(
    IterableDataset[tuple[torch.FloatTensor, torch.FloatTensor]]
):

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
    """

    def __init__(
        self,
        regions_of_interest: pd.DataFrame,
        collection: Union[str, Sequence[str], Path, Sequence[Path], BigWigCollection],
        reference_genome_path: Path,
        sequence_length: int = 1000,
        center_bin_to_predict: Optional[int] = 200,
        window_size: int = 1,
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
    ):
        super().__init__()
        self._dataset = BigWigDataset(
            regions_of_interest=regions_of_interest,
            collection=collection,
            reference_genome_path=reference_genome_path,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            batch_size=batch_size,
            super_batch_size=super_batch_size,
            batches_per_epoch=batches_per_epoch,
            maximum_unknown_bases_fraction=maximum_unknown_bases_fraction,
            sequence_encoder=sequence_encoder,
            file_extensions=file_extensions,
            crawl=crawl,
            first_n_files=first_n_files,
            position_sampler_buffer_size=position_sampler_buffer_size,
        )
        if window_size and window_size > 1:
            self.average_pool_target: Optional[torch.nn.AvgPool1d] = torch.nn.AvgPool1d(
                kernel_size=window_size, stride=window_size
            )
        else:
            self.average_pool_target = None

    def __iter__(self) -> Iterator[tuple[torch.FloatTensor, torch.FloatTensor]]:
        iter(self._dataset)
        return self

    def __next__(self) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        sequences, target = next(self._dataset)
        target = torch.as_tensor(target, device="cuda")
        sequences = torch.FloatTensor(sequences)
        if self.average_pool_target:
            target = self.average_pool_target(target)
        return sequences, target
