import time
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import pyBigWig
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from bigwig_loader import config
from bigwig_loader.sampler.genome_sampler import GenomicSequenceBatchSampler
from bigwig_loader.sampler.position_sampler import RandomPositionSampler


class PyBigWigDataset:
    """
    Dataset over FASTA files and BigWig profiles.
    Args:
        regions_of_interest: pandas dataframe with intervals within the chromosomes to
            sample from. These intervals are not intervals of sequence_length
            but regions of the chromosome that were selected up front. For
            instance regions in which called peaks are located or a threshold
            value is exceeded.
        bigwig_path: path to BigWig Directory or list of BigWig files.
        reference_genome_path: path to fasta file containing the reference genome.
        sequence_length: number of base pairs in input sequence
        center_bin_to_predict: if given, only do prediction on a central window. Should be
            smaller than or equal to sequence_length. If not given will be the same as
            sequence_length.
        window_size: used to down sample the resolution of the target from sequence_length
        batch_size: batch size
        batches_per_epoch: because the length of an epoch is slightly arbitrary here,
            the number of batches can be set by hand. If not the number of batches per
            epoch will be (totol number of bases in combined intervals) // sequence_length // batch_size
        maximum_unknown_bases_fraction: maximum number of bases in an input sequence that
            is unknown.
        sequence_encoder: encoder to apply to the sequence. Default: bigwig_loader.util.onehot_sequences
    """

    def __init__(
        self,
        regions_of_interest: pd.DataFrame,
        bigwig_path: Path,
        reference_genome_path: Path,
        sequence_length: int = 1000,
        center_bin_to_predict: Optional[int] = 200,
        batch_size: int = 256,
        batches_per_epoch: Optional[int] = None,
        maximum_unknown_bases_fraction: float = 0.1,
        sequence_encoder: Optional[Union[Callable, str]] = "onehot",
    ):
        super().__init__()
        self.regions_of_interest = regions_of_interest
        self.bigwig_path = bigwig_path
        self.reference_genome_path = reference_genome_path
        self.sequence_length = sequence_length
        self.center_bin_to_predict = center_bin_to_predict
        self.batch_size = batch_size
        self.batches_per_epoch = (
            batches_per_epoch
            or regions_of_interest.index[-1] // sequence_length // batch_size
        )
        self._n = 0
        self.maximum_unknown_bases_fraction = maximum_unknown_bases_fraction
        self.sequence_encoder = sequence_encoder

    def _setup_genome(self):
        position_sampler = RandomPositionSampler(
            regions_of_interest=self.regions_of_interest
        )
        self.genome = GenomicSequenceBatchSampler(
            self.reference_genome_path,
            sequence_length=self.sequence_length,
            position_sampler=position_sampler,
            batch_size=self.batch_size,
            maximum_unknown_bases_fraction=self.maximum_unknown_bases_fraction,
            encoder=self.sequence_encoder,
        )

    def _setup_bigwig_collection(self):
        # self.bigwig_collection = BigWigCollection(self.bigwig_path)
        bigwig_paths = list(self.bigwig_path.glob("*.bigWig")) + list(
            self.bigwig_path.glob("*.bw")
        )
        self.bigwigs = [pyBigWig.open(str(bw)) for bw in bigwig_paths]

    def __iter__(self):
        self._setup_genome()
        self._setup_bigwig_collection()
        self._n = 0
        return self

    def __next__(self):
        if self._n < self.batches_per_epoch:
            self._n += 1
            positions, sequences = self.genome.get_batch()
            chromosomes, center = zip(*positions)
            start = np.array(center) - (self.center_bin_to_predict // 2)
            end = start + self.center_bin_to_predict
            target = self.get_target_batch(chromosomes, start, end)
            return sequences, target
        raise StopIteration

    def get_target_batch(self, chromosomes, starts, ends):
        return np.stack(
            [
                bw.values(chrom, start, end, numpy=True)
                for bw in self.bigwigs
                for chrom, start, end in zip(chromosomes, starts, ends)
            ]
        )


class PytorchPyBigWigDataset(IterableDataset):

    """
    Pytorch IterableDataset over FASTA files and BigWig profiles.
    Args:
        regions_of_interest: pandas dataframe with intervals within the chromosomes to
            sample from. These intervals are not intervals of sequence_length
            but regions of the chromosome that were selected up front. For
            instance regions in which called peaks are located or a threshold
            value is exceeded.
        bigwig_path: path to BigWig Directory or list of BigWig files.
        reference_genome_path: path to fasta file containing the reference genome.
        sequence_length: number of base pairs in input sequence
        center_bin_to_predict: if given, only do prediction on a central window. Should be
            smaller than or equal to sequence_length. If not given will be the same as
            sequence_length.
        window_size: used to down sample the resolution of the target from sequence_length
        batch_size: batch size
        batches_per_epoch: because the length of an epoch is slightly arbitrary here,
            the number of batches can be set by hand. If not the number of batches per
            epoch will be (totol number of bases in combined intervals) // sequence_length // batch_size
        maximum_unknown_bases_fraction: maximum number of bases in an input sequence that
            is unknown.
        sequence_encoder: encoder to apply to the sequence. Default: bigwig_loader.util.onehot_sequences
    """

    def __init__(
        self,
        regions_of_interest: pd.DataFrame,
        bigwig_path: Path,
        reference_genome_path: Path,
        sequence_length: int = 1000,
        center_bin_to_predict: Optional[int] = 200,
        window_size: int = 1,
        batch_size: int = 256,
        batches_per_epoch: Optional[int] = None,
        maximum_unknown_bases_fraction: float = 0.1,
        sequence_encoder: Optional[Union[Callable, str]] = "onehot",
    ):
        super().__init__()
        self._dataset = PyBigWigDataset(
            regions_of_interest=regions_of_interest,
            bigwig_path=bigwig_path,
            reference_genome_path=reference_genome_path,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            maximum_unknown_bases_fraction=maximum_unknown_bases_fraction,
            sequence_encoder=sequence_encoder,
        )
        if window_size and window_size > 1:
            self.average_pool_target = torch.nn.AvgPool1d(
                kernel_size=window_size, stride=window_size
            )
        else:
            self.average_pool_target = None

    def __iter__(self):
        iter(self._dataset)
        return self

    def __next__(self):
        sequences, target = next(self._dataset)
        target = torch.from_numpy(target).float()
        sequences = torch.FloatTensor(sequences)
        if self.average_pool_target:
            target = self.average_pool_target(target)
        return sequences, target


def run():
    train_regions = pd.read_csv("train_regions.tsv", sep="\t")
    print("Loading from:", config.bigwig_dir)

    dataset = PytorchPyBigWigDataset(
        regions_of_interest=train_regions,
        bigwig_path=config.bigwig_dir,
        reference_genome_path=config.reference_genome,
        sequence_length=1000,
        center_bin_to_predict=1000,
        window_size=1,
        batch_size=256,
        batches_per_epoch=20,
        maximum_unknown_bases_fraction=0.1,
        sequence_encoder="onehot",
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=8)

    elapsed = []
    for i, (X, Y) in enumerate(loader):
        if i == 0:
            start = time.perf_counter()
            continue
        end = time.perf_counter()
        elapsed.append(end - start)
        start = end
        # if not i % 100:
        print("Seconds per batch:", sum(elapsed) / i)
        print(Y.shape)


if __name__ == "__main__":
    run()
