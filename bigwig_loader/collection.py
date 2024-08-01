import logging
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd

from bigwig_loader.bigwig import BigWig
from bigwig_loader.decompressor import Decoder
from bigwig_loader.intervals_to_values import intervals_to_values
from bigwig_loader.memory_bank import MemoryBank
from bigwig_loader.memory_bank import create_memory_bank
from bigwig_loader.merge_intervals import merge_interval_dataframe
from bigwig_loader.path import interpret_path
from bigwig_loader.path import map_path_to_value
from bigwig_loader.searchsorted import interval_searchsorted
from bigwig_loader.subtract_intervals import subtract_interval_dataframe
from bigwig_loader.util import chromosome_sort


class BigWigCollection:
    """
    Wrapper around pyBigWig wrapping a collection of BigWig files.
    args:
        bigwig_path: path to BigWig Directory or list of BigWig files.
        file_extensions: only used when looking in directories for BigWig files.
            All files with these extensions are assumed to be BigWIg files:
            default: (".bigWig", ".bw").
        crawl: to walk the directory tree or not. If False, subdirectories are
            ignored.
        scale: Optional, dictionary with scaling factors for each BigWig file.
            If None, no scaling is done. Keys can be (partial) file paths. See
            bigwig_loader.path.match_key_to_path for more information about how
            dict keys are mapped to paths.
        first_n_files: Optional, only consider the first n files (after sorting).
            Handy for debugging.
        pinned_memory_size: size of pinned memory used to load compressed data to.
        use_cufile: whether to use kvikio cuFile to directly load data from file to
            GPU memory.

    """

    def __init__(
        self,
        bigwig_path: Union[str, Sequence[str], Path, Sequence[Path]],
        file_extensions: Sequence[str] = (".bigWig", ".bw"),
        crawl: bool = True,
        scale: Optional[dict[Union[str | Path], Any]] = None,
        first_n_files: Optional[int] = None,
        pinned_memory_size: int = 10000,
        use_cufile: bool = True,
    ):
        self._use_cufile = use_cufile
        self.bigwig_paths = sorted(
            interpret_path(bigwig_path, file_extensions=file_extensions, crawl=crawl)
        )[:first_n_files]

        scale = scale or {}
        self._scaling_factors = [
            map_path_to_value(path, value_dict=scale, default=1)
            for path in self.bigwig_paths
        ]

        self.bigwigs = [
            BigWig(path, id=i, scale=scaling_factor, use_cufile=use_cufile)
            for i, (path, scaling_factor) in enumerate(
                zip(self.bigwig_paths, self._scaling_factors)
            )
        ]

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
        if "scaling_factors_cupy" in self.__dict__:
            del self.__dict__["scaling_factors_cupy"]

    @cached_property
    def decoder(self) -> Decoder:
        return Decoder(
            max_rows_per_chunk=self.max_rows_per_chunk,
            max_uncompressed_chunk_size=self.max_rows_per_chunk * 12 + 24,
            chromosome_offsets=self.local_chrom_ids_to_offset_matrix,
        )

    @cached_property
    def memory_bank(self) -> MemoryBank:
        return create_memory_bank(
            nbytes=self.pinned_memory_size, elastic=True, use_cufile=self._use_cufile
        )

    @cached_property
    def scaling_factors_cupy(self) -> cp.ndarray:
        return cp.asarray(self._scaling_factors, dtype=cp.float32).reshape(
            1, len(self._scaling_factors), 1
        )

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

    def batch_load(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
        memory_bank: MemoryBank,
    ) -> tuple[
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
    ]:
        memory_bank.reset()

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
                bigwig.store.file_handle,
                offsets,
                sizes,
                skip_bytes=2,
            )

        abs_end = cp.asarray(abs_end, dtype=cp.uint32)
        abs_start = cp.asarray(abs_start, dtype=cp.uint32)

        # bring the gpu
        bigwig_ids = cp.asarray(bigwig_ids, dtype=cp.uint32)
        n_chunks_per_bigwig = cp.asarray(n_chunks_per_bigwig, dtype=cp.uint32)

        comp_chunk_pointers, compressed_chunk_sizes = self.memory_bank.to_gpu()
        _, start_data, end_data, value_data, n_rows_for_chunks = self.decoder.decode(
            comp_chunk_pointers, compressed_chunk_sizes, bigwig_ids=bigwig_ids
        )

        return (
            abs_start,
            abs_end,
            start_data,
            end_data,
            value_data,
            n_chunks_per_bigwig,
            n_rows_for_chunks,
        )

    def batch_searchsorted(
        self,
        start_data: cp.ndarray,
        end_data: cp.ndarray,
        abs_start: cp.ndarray,
        abs_end: cp.ndarray,
        n_chunks_per_bigwig: cp.ndarray,
        n_rows_for_chunks: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        bigwig_starts = cp.pad(cp.cumsum(n_chunks_per_bigwig), (1, 0))
        chunk_starts = cp.pad(cp.cumsum(n_rows_for_chunks), (1, 0))
        bigwig_starts = chunk_starts[bigwig_starts]
        sizes = bigwig_starts[1:] - bigwig_starts[:-1]

        sizes = sizes.astype(cp.uint32)

        return interval_searchsorted(
            array_start=start_data,
            array_end=end_data,
            query_starts=abs_start,
            query_ends=abs_end,
            sizes=sizes,
            absolute_indices=True,
        )

    def get_batch(
        self,
        chromosomes: Union[Sequence[str], npt.NDArray[np.generic]],
        start: Union[Sequence[int], npt.NDArray[np.int64]],
        end: Union[Sequence[int], npt.NDArray[np.int64]],
        window_size: int = 1,
        out: Optional[cp.ndarray] = None,
    ) -> cp.ndarray:
        (
            abs_start,
            abs_end,
            start_data,
            end_data,
            value_data,
            n_chunks_per_bigwig,
            n_rows_for_chunks,
        ) = self.batch_load(
            chromosomes=chromosomes, start=start, end=end, memory_bank=self.memory_bank
        )

        found_starts, found_ends = self.batch_searchsorted(
            start_data,
            end_data,
            abs_start,
            abs_end,
            n_chunks_per_bigwig,
            n_rows_for_chunks,
        )

        print("sssssss")
        print(found_starts, found_ends)

        if out is None:
            sequence_length = (end[0] - start[0]) // window_size
            out = self._get_out_tensor(len(start), sequence_length)

        intervals_to_values(
            array_start=start_data,
            array_end=end_data,
            array_value=value_data,
            found_starts=found_starts,
            found_ends=found_ends,
            query_starts=abs_start,
            query_ends=abs_end,
            window_size=window_size,
            out=out,
        )
        batch = cp.transpose(out, (1, 0, 2))
        batch *= self.scaling_factors_cupy
        return batch

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
        blacklist: Optional[pd.DataFrame] = None,
        blacklist_buffer: int = 0,
        padding: int = 0,
        merge: bool = False,
        threshold: float = 0.0,
        merge_allow_gap: int = 0,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        """
        Get Intervals from the collection of intervals. This function
        in turn calls the intervals method of the BigWig class for each
        BigWig file in the collection and concatenates, and if wanted merges,
        the results. If the "scale" argument was used when the collection was
        created, the values are scaled accordingly and the threshold is applied
        on the scaled values.
        Args:
            include_chromosomes: list of chromosome, "standard" or "all" (default).
            exclude_chromosomes: list of chromosomes you want to exclude
            blacklist: pandas dataframe of intervals that you want to
                exclude from the result.
            blacklist_buffer: default 0. Buffer around blacklist intervals to
                exclude.
            padding: padding around intervals, before start and after end.
                Default 0.
            threshold: only return intervals of which the value exceeds
                this threshold.
            merge: whether to merge intervals that are directly following
                each other. The value will be the max value of the merged
                intervals.
            merge_allow_gap: default 0. Allow intervals seperated by size
                merge_allow_gap bases to still be merged.
            batch_size: number of intervals processed at once.
        Returns: pandas dataframe of intervals (chrom, start, end, value)

        """
        logging.info("Collecting intervals from BigWig files.")
        interval_for_all_bigwigs = []
        n_bigwigs = len(self.bigwigs)
        for i, bw in enumerate(self.bigwigs):
            logging.info(
                f"Getting intervals for BigWig file {i}/{n_bigwigs}: {bw.path}"
            )
            interval_for_all_bigwigs.append(
                bw.intervals(
                    include_chromosomes=include_chromosomes,
                    exclude_chromosomes=exclude_chromosomes,
                    blacklist=blacklist,
                    blacklist_buffer=blacklist_buffer,
                    threshold=threshold,
                    merge=merge,
                    merge_allow_gap=merge_allow_gap,
                    memory_bank=self.memory_bank,
                    decoder=self.decoder,
                    batch_size=batch_size,
                )
            )

        intervals = pd.concat(interval_for_all_bigwigs)
        if padding:
            intervals["start"] = intervals["start"].values.clip(padding) - padding  # type: ignore
            intervals["end"] = intervals["end"].values + padding  # type: ignore
        if merge:
            logging.info(
                "Merging intervals from different BigWig Files into one interval list."
            )
            intervals = merge_interval_dataframe(
                intervals, is_sorted=False, allow_gap=merge_allow_gap
            )
        if blacklist is not None:
            logging.info("Subtracting blacklisted regions.")
            intervals = subtract_interval_dataframe(
                intervals=intervals, blacklist=blacklist, buffer=blacklist_buffer
            )
        self.memory_bank.shrink()

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
