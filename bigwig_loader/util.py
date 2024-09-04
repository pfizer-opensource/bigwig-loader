from typing import Iterable
from typing import Optional
from typing import Sequence

import cupy as cp
import numpy as np
import pandas as pd
from natsort import natsort_keygen
from natsort import natsorted


def get_standard_chromosomes(
    exclude: Optional[Iterable[str]] = ("chrX", "chrY", "chrM")
) -> list[str]:
    exclude = exclude or set()
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    # Doing like this instead of with set to preserve nice order.
    return [chrom for chrom in chromosomes if chrom not in exclude]


def sort_intervals(intervals: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    if inplace:
        intervals.sort_values(by=["chrom", "start"], key=natsort_keygen(), inplace=True)
        return intervals
    else:
        return intervals.sort_values(
            by=["chrom", "start"], key=natsort_keygen(), inplace=False
        )


def make_cumulative_index_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    intervals.reset_index(drop=True, inplace=True)
    intervals.index = (
        (intervals["end"] - intervals["start"]).cumsum().shift().fillna(0).astype(int)  # type: ignore
    )
    return intervals


_string_to_encoding = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "R": [0.5, 0, 0.5, 0],
    "Y": [0, 0.5, 0, 0.5],
    "K": [0, 0, 0.5, 0.5],
    "M": [0.5, 0.5, 0, 0],
    "S": [0, 0.5, 0.5, 0],
    "W": [0.5, 0, 0, 0.5],
    "B": [0, 0.333, 0.333, 0.333],
    "D": [0.333, 0, 0.333, 0.333],
    "H": [0.333, 0.333, 0, 0.333],
    "V": [0.333, 0.333, 0.333, 0],
    "N": [0.25, 0.25, 0.25, 0.25],
}


_standard_bases = {"A", "C", "G", "T"}

encoding_array = np.zeros((26, 4), dtype=cp.float32)
for letter, encoding in _string_to_encoding.items():
    index = ord(letter) - 65  # Calculate the index based on the ASCII value
    encoding_array[index] = encoding


def fraction_non_standard(sequence: str) -> float:
    standard_bases = _standard_bases
    n_non_standard = 0
    for base in sequence:
        if base not in standard_bases:
            n_non_standard += 1
    return n_non_standard / len(sequence)


def onehot_sequences(sequences: Sequence[str]) -> list[list[list[float]]]:
    return [[_string_to_encoding[base] for base in sequence] for sequence in sequences]


def chromosome_sort(chromosomes: Iterable[str]) -> list[str]:
    chromosomes = set(chromosomes)
    standard = set(get_standard_chromosomes(exclude=None))
    standard_present = chromosomes.intersection(standard)
    rest = chromosomes - standard_present
    return natsorted(standard_present) + natsorted(rest)  # type: ignore


def onehot_sequences_cupy(sequences: Sequence[str]) -> cp.ndarray:
    n_sequences = len(sequences)
    sequence_length = len(sequences[0])
    cupy_encoding_aray = cp.asarray(encoding_array)

    sequence = "".join(sequences)

    # Convert DNA sequence to bytes (ASCII values)
    # Subtract 65 from the byte array to get indices corresponding to encoding_array
    indices = cp.frombuffer(sequence.encode("ascii"), dtype=cp.int8) - 65

    one_hot_encoded = cupy_encoding_aray[indices]
    return one_hot_encoded.reshape(n_sequences, sequence_length, 4)


if __name__ == "__main__":
    onehot = onehot_sequences_cupy(["AAAAATTTTACGT", "CAGAATTGTACGT"])
    print(onehot)
