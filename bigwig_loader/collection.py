import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd
from upath import UPath

from bigwig_loader.bigwig import BigWig
from bigwig_loader.gpu_decompressor import Decoder
from bigwig_loader.intervals_to_values_gpu import intervals_to_values
from bigwig_loader.memory_bank import MemoryBank
from bigwig_loader.merge_intervals import merge_interval_dataframe
from bigwig_loader.util import chromosome_sort


class BigWigCollection:
    """
    Wrapper around pyBigWig wrapping a collection of BigWig files.
    args:
        bigwig_path: path to BigWig Directory or list of BigWig files.
        file_extensions: only used when looking in directories for BigWig files.
            All files with these extensions are assumed to be BigWIg files:
            default: (".bigWig", ".bw").
        walk: to walk the directory tree or not. If False, subdirectories are
            ignored.
        first_n_files: Optional, only consider the first n files (after sorting).
            Handy for debugging.
    """

    def __init__(
        self,
        bigwig_path: Union[str, Sequence[str], Path, Sequence[Path]],
        file_extensions: Sequence[str] = (".bigWig", ".bw"),
        crawl: bool = True,
        first_n_files: Optional[int] = None,
        pinned_memory_size: int = 10000,
    ):
        self.bigwig_paths = sorted(
            interpret_path(bigwig_path, file_extensions=file_extensions, crawl=crawl)
        )[:first_n_files]
        self.bigwigs = [BigWig(path, id=i) for i, path in enumerate(self.bigwig_paths)]

        (
            self.chromosome_offset_dict,
            self.local_chrom_ids_to_offset_matrix,
        ) = self.create_global_position_system()

        self.max_rows_per_chunk = max(
            [bigwig.max_rows_per_chunk for bigwig in self.bigwigs]
        )
        self.pinned_memory_size = pinned_memory_size
        self._out: cp.ndarray = cp.zeros((len(self), 1, 1), dtype=cp.float32)

        self.run_indexing()

    def run_indexing(self) -> None:
        for bigwig in self.bigwigs:
            bigwig.run_indexing(
                chromosome_offsets=self.local_chrom_ids_to_offset_matrix[bigwig.id]
            )

    def __len__(self) -> int:
        return len(self.bigwigs)

    def reset_gpu(self) -> None:
        """
        Remove all gpu arrays from the previously used device and recreate when necessary on
        current device. This is useful when training is done on multiple gpus and the arrays
        need to be recreated on the new gpu.
        """

        self._out = cp.zeros((len(self), 1, 1), dtype=cp.float32)
        if "decoder" in self.__dict__:
            del self.__dict__["decoder"]
        if "memory_bank" in self.__dict__:
            del self.__dict__["memory_bank"]

    @cached_property
    def decoder(self) -> Decoder:
        return Decoder(
            max_rows_per_chunk=self.max_rows_per_chunk,
            max_uncompressed_chunk_size=self.max_rows_per_chunk * 12 + 24,
            chromosome_offsets=self.local_chrom_ids_to_offset_matrix,
        )

    @cached_property
    def memory_bank(self) -> MemoryBank:
        return MemoryBank(nbytes=self.pinned_memory_size, elastic=True)

    def _get_out_tensor(self, batch_size: int, sequence_length: int) -> cp.ndarray:
        """Resuses a reserved tensor if possible (when out shape is constant),
        otherwise creates a new one.
        args:
            batch_size: batch size
            sequence_length: length of genomic sequence
         returns:
            tensor of shape (number of bigwig files, batch_size, sequence_length)
        """

        shape = (len(self), batch_size, sequence_length)
        if self._out.shape != shape:
            self._out = cp.zeros(shape, dtype=cp.float32)
        return self._out

    def get_batch(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
        window_size: int = 1,
        out: Optional[cp.ndarray] = None,
    ) -> cp.ndarray:
        self.memory_bank.reset()

        if (end[0] - start[0]) % window_size:
            raise ValueError(
                f"Sequence length {end[0] - start[0]} is not divisible by window size {window_size}"
            )

        if out is None:
            sequence_length = (end[0] - start[0]) // window_size
            out = self._get_out_tensor(len(start), sequence_length)

        abs_start = self.make_positions_global(chromosomes, start)
        abs_end = self.make_positions_global(chromosomes, end)

        n_chunks_per_bigwig = []
        bigwig_ids = []
        for bigwig in self.bigwigs:
            offsets, sizes = bigwig.get_batch_offsets_and_sizes_with_global_positions(
                abs_start, abs_end
            )
            n_chunks_per_bigwig.append(len(offsets))
            bigwig_ids.extend([bigwig.id] * len(offsets))
            # read chunks into preallocated memory
            self.memory_bank.add_many(
                bigwig.store._fh,
                offsets,
                sizes,
                skip_bytes=2,
            )

        # bring the gpu
        bigwig_ids = cp.asarray(bigwig_ids, dtype=cp.uint32)
        gpu_byte_array, comp_chunks, compressed_chunk_sizes = self.memory_bank.to_gpu()
        _, start, end, value, n_rows_for_chunks = self.decoder.decode(
            gpu_byte_array, comp_chunks, compressed_chunk_sizes, bigwig_ids=bigwig_ids
        )

        logging.debug(
            f"decompressed array sizes {len(start), len(end), len(value), len(n_rows_for_chunks), len(bigwig_ids)}"
        )
        logging.debug(
            f"Cupy default memory pool:{ cp.get_default_memory_pool().used_bytes() / 1024} kB"
        )

        chunk_row_numbers = cp.pad(cp.cumsum(n_rows_for_chunks), (1, 0))
        i = 0
        for n_chunks, partial_out in zip(n_chunks_per_bigwig, out):
            bigwig_end = i + n_chunks

            row_number_start = chunk_row_numbers[i]
            row_number_end = chunk_row_numbers[bigwig_end]

            intervals_to_values(
                track_starts=start[row_number_start:row_number_end],
                track_ends=end[row_number_start:row_number_end],
                track_values=value[row_number_start:row_number_end],
                query_starts=cp.asarray(abs_start, dtype=cp.uint32),
                query_ends=cp.asarray(abs_end, dtype=cp.uint32),
                window_size=window_size,
                out=partial_out,
            )
            i = bigwig_end
        return cp.transpose(out, (1, 0, 2))

    def make_positions_global(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        positions: Union[Sequence[int], npt.NDArray[np.int64]],
    ) -> npt.NDArray[np.int64]:
        """
        Take a set of positions and corresponding chromosomes
        and transform those positions into a coordinates that
        run over all chromosomes by adding specific offsets for
        chromosomes.

        Args:
            chromosomes: chromosome keys like "chr1, chr2...."
            positions: numpy array of integer positions on the chromosomes

        Returns:
            numpy array with "global" positions

        """
        offsets = np.array(
            [self.chromosome_offset_dict[chrom] for chrom in chromosomes]
        )
        return positions + offsets

    def intervals(
        self,
        include_chromosomes: Union[Literal["all", "standard"], Sequence[str]] = "all",
        exclude_chromosomes: Optional[Sequence[str]] = None,
        merge: bool = False,
        threshold: float = 0.0,
        merge_allow_gap: int = 0,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        intervals = pd.concat(
            [
                bw.intervals(
                    include_chromosomes=include_chromosomes,
                    exclude_chromosomes=exclude_chromosomes,
                    threshold=threshold,
                    merge=merge,
                    merge_allow_gap=merge_allow_gap,
                    memory_bank=self.memory_bank,
                    decoder=self.decoder,
                    batch_size=batch_size,
                )
                for bw in self.bigwigs
            ]
        )
        if merge:
            intervals = merge_interval_dataframe(
                intervals, is_sorted=False, allow_gap=merge_allow_gap
            )
        return intervals

    def chromosomes(
        self,
        include_chromosomes: Union[Literal["all", "standard"], Sequence[str]] = "all",
        exclude_chromosomes: Optional[Sequence[str]] = None,
    ) -> set[str]:
        """
        Get a list of all the chromosomes present in files.

        Args:
            include_chromosomes: chromosome keys you want to include.
                Can alternatively be "all" or "standard".
            exclude_chromosomes: Optional set of chromosomes to exclude.

        Returns: a list of chromosomes over all BigWig files in the format:
            chr1_kidney, chr3_heart etc...

        """
        return {
            f"{chrom}_{path.stem}"
            for bigwig, path in zip(self.bigwigs, self.bigwig_paths)
            for chrom in bigwig.chromosomes(
                include_chromosomes=include_chromosomes,
                exclude_chromosomes=exclude_chromosomes,
            )
        }

    def get_chromosomes_present_in_all_files(
        self,
        include_chromosomes: Union[Literal["all", "standard"], Sequence[str]] = "all",
        exclude_chromosomes: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """
        Get the subset of chromosome keys (i.e. "chr1", "chr4"...) that
        is present in every single BigWig file. The intersection of sets
        of chromosome keys over all the files.
        Args:
            include_chromosomes: chromosome keys you want to include.
                Can alternatively be "all" or "standard".
            exclude_chromosomes: Optional set of chromosomes to exclude.

        Returns:
            A list of chromosomes in a nicely sorted order.

        """
        return chromosome_sort(
            set.intersection(
                *[
                    set(
                        bigwig.chromosomes(
                            include_chromosomes=include_chromosomes,
                            exclude_chromosomes=exclude_chromosomes,
                        )
                    )
                    for bigwig in self.bigwigs
                ]
            )
        )

    def create_global_position_system(
        self,
    ) -> tuple[dict[str, int], npt.NDArray[np.int64]]:
        """
        Bigwig files internally use chrom_ids which are integers mapping to the different chromosomes.
        The problem is that this mapping to the actual chromosomes can actually be different from file
        to file. For performance reasons we want to decompress chunks from different files at the same
        time and while we are at it convert combinations of (chrom_id, position) to a positions system
        of a single integer that runs over all the chromosomes. For this we need to create a matrix
        local_chrom_ids_to_offset of size (n_files x n_chrom_ids) that we can use later on to add
        offsets to positions after a quick indexing operation:

            offsets = local_chrom_ids_to_offset[file_ids, chrom_ids]

        Returns:
            Tuple(Dict mapping chromosome keys to offsets like {"chr1": 200, "chr2": 250, },
                  np.ndarray size (n_files x n_chrom_ids) as described.

        """
        # find all chromosome keys (i.e. chr1, chr2...)
        unique_chromosomes = chromosome_sort(
            {
                chromosome
                for bigwig in self.bigwigs
                for chromosome in bigwig.chromosomes()
            }
        )
        global_chromosome_ids = {key: i for i, key in enumerate(unique_chromosomes)}
        max_local_chromosome_id = max(
            [
                chrom_id
                for bigwig in self.bigwigs
                for chrom_id in bigwig.chrom_to_chrom_id.values()
            ]
        )
        chrom_id_mapping_matrix = np.zeros(
            shape=(len(self.bigwigs), max_local_chromosome_id + 1), dtype=np.uint32
        )

        chromosome_sizes = np.array(
            [
                max(  # type: ignore
                    [
                        bigwig.chromosome_sizes.get(chromosome, 0)
                        for bigwig in self.bigwigs
                    ]
                )
                for chromosome in unique_chromosomes
            ]
        )
        chromosome_offsets = np.pad(np.cumsum(chromosome_sizes), (1, 0))
        chromosome_offset_dict = {
            key: size for key, size in zip(unique_chromosomes, chromosome_offsets)
        }
        if chromosome_offsets[-1] > 2**32:
            # exceeding this would be a problem as all the positions
            # in bigwig files are encoded in 32 bits. Making a global
            # base position that exceeds this number would be a problem.
            logging.warning(
                "The sum of sizes of the chromosomes exceeds the size of uint32"
            )
        for bigwig in self.bigwigs:
            for key, chrom_id in bigwig.chrom_to_chrom_id.items():
                chrom_id_mapping_matrix[bigwig.id, chrom_id] = global_chromosome_ids[
                    key
                ]
        local_chrom_ids_to_offset = chromosome_offsets[chrom_id_mapping_matrix]
        return chromosome_offset_dict, local_chrom_ids_to_offset


def get_bigwig_files_from_path(
    path: Path, file_extensions: Iterable[str] = (".bigWig", ".bw"), crawl: bool = True
) -> list[Path]:
    """
    For a path object, get all bigwig files. If the path is directly pointing
    to a bigwig file, a list with just that file is returned. In case of a
    directory all files are gathered with file extensions that are part of
    file_extensions.
    Args:
        path: or upath.Path object. Either directory or BigWig file
        file_extensions: used to find all BigWig files in a directory.
            Default: (".bigWig", ".bw")
        crawl: whether to find BigWig files in subdirectories. Default: True.

    Returns: list of paths to BigWig files.

    """
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")
    elif path.is_dir():
        if crawl:
            pattern = "**/*"
        else:
            pattern = "*"
        return [
            file
            for extension in file_extensions
            for file in path.glob(f"{pattern}{extension}")
        ]
    return [path]


def interpret_path(
    bigwig_path: Union[
        Union[str, "os.PathLike[Any]"], Iterable[Union[str, "os.PathLike[Any]"]]
    ],
    file_extensions: Iterable[str] = (".bigWig", ".bw"),
    crawl: bool = True,
) -> list[Path]:
    """
    Get all bigwig files for a path. Also excepts strings.
    If the path is directly pointing to a bigwig file, a list with just that
    file is returned. In case of a directory all files are gathered with file
    extensions that are part of file_extensions.
    Args:
        bigwig_path: str, Path or upath.Path object. Either directory or BigWig file
        file_extensions: used to find all BigWig files in a directory.
            Default: (".bigWig", ".bw")
        crawl: whether to find BigWig files in subdirectories. Default: True.

    Returns: list of paths to BigWig files.

    """
    if isinstance(bigwig_path, str) or isinstance(bigwig_path, os.PathLike):
        return get_bigwig_files_from_path(
            UPath(bigwig_path), file_extensions=file_extensions, crawl=crawl
        )

    elif isinstance(bigwig_path, Iterable):
        return [
            path
            for element in bigwig_path
            for path in interpret_path(
                element, file_extensions=file_extensions, crawl=crawl
            )
        ]
    raise ValueError(
        f"Can not interpret {bigwig_path} as path or a collection of paths."
    )
