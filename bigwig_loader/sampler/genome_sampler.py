from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import fsspec
from pyfaidx import Fasta

from bigwig_loader.util import fraction_non_standard
from bigwig_loader.util import onehot_sequences

ENCODERS: dict[str, Callable[[Sequence[str]], Any]] = {"onehot": onehot_sequences}


class GenomicSequenceSampler:
    def __init__(
        self,
        reference_genome_path: Path,
        sequence_length: int,
        position_sampler: Iterator[tuple[str, int]],
        maximum_unknown_bases_fraction: float = 0.1,
    ):
        self.reference_genome_path = reference_genome_path
        self.sequence_length = sequence_length
        self.position_sampler = position_sampler
        self.maximum_unknown_bases_fraction = maximum_unknown_bases_fraction
        self.genome = Fasta(fsspec.open(self.reference_genome_path))
        self._half_sequence_length = self.sequence_length // 2

    def __iter__(self) -> Iterator[tuple[str, int, str]]:
        for chromosome, center in self.position_sampler:
            start = center - self._half_sequence_length
            end = start + self.sequence_length
            sequence = self._get_sequence(chromosome, start, end)
            if not sequence:
                continue
            if len(sequence) != self.sequence_length:
                continue
            if fraction_non_standard(sequence) > self.maximum_unknown_bases_fraction:
                continue
            yield chromosome, center, sequence

    def _get_sequence(
        self, chrom: str, start: int, end: int, strand: Literal["+", "-"] = "+"
    ) -> Any:
        return self.genome[chrom][start:end].seq


class GenomicSequenceBatchSampler:
    """Batched version of GenomicSequenceSampler."""

    def __init__(
        self,
        reference_genome_path: Path,
        sequence_length: int,
        position_sampler: Iterator[tuple[str, int]],
        batch_size: int,
        maximum_unknown_bases_fraction: float = 0.1,
        encoder: Optional[
            Union[Callable[[Sequence[str]], Any], Literal["onehot"]]
        ] = "onehot",
    ):
        self._genomic_sequence_sampler = GenomicSequenceSampler(
            reference_genome_path=reference_genome_path,
            sequence_length=sequence_length,
            position_sampler=position_sampler,
            maximum_unknown_bases_fraction=maximum_unknown_bases_fraction,
        )
        self.batch_size = batch_size
        self.encoder = self._select_encoder(encoder)

    def __iter__(self) -> Iterator[tuple[list[str], list[int], list[Any]]]:
        return self

    def __next__(self) -> tuple[list[str], list[int], list[Any]]:
        return self.get_batch()

    def get_batch(self) -> tuple[list[str], list[int], list[Any]]:
        chromosomes = []
        centers = []
        sequences = []

        for _, (chromosome, center, sequence) in zip(
            range(self.batch_size), self._genomic_sequence_sampler
        ):
            chromosomes.append(chromosome)
            centers.append(center)
            sequences.append(sequence)
        if self.encoder is not None:
            sequences = self.encoder(sequences)
        return chromosomes, centers, sequences

    def _select_encoder(
        self,
        encoder: Optional[Union[Callable[[Sequence[str]], Any], Literal["onehot"]]],
    ) -> Optional[Callable[[Sequence[str]], Any]]:
        if encoder is None:
            return None
        if callable(encoder):
            return encoder
        if isinstance(encoder, str) and encoder in ENCODERS:
            return ENCODERS[encoder]
        raise ValueError(
            f"{encoder} is not a valid encoder. It is not listed in {list(ENCODERS.keys())} and also not a Callable."
        )


def register_encoder(encoder_id: str, encoder: Callable[[Sequence[str]], Any]) -> None:
    ENCODERS[encoder_id] = encoder
