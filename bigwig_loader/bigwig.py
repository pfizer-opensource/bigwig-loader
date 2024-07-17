import zlib
from pathlib import Path
from random import sample
from typing import BinaryIO
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd
from ncls import NCLS

from bigwig_loader.gpu_decompressor import Decoder
from bigwig_loader.memory_bank import MemoryBank
from bigwig_loader.merge_intervals import merge_intervals
from bigwig_loader.parser import BBIHeader
from bigwig_loader.parser import ChromosomeTreeHeader
from bigwig_loader.parser import ChromosomeTreeNode
from bigwig_loader.parser import RTreeIndexHeader
from bigwig_loader.parser import RTreeNode
from bigwig_loader.parser import TotalSummary
from bigwig_loader.parser import WIGSectionHeader
from bigwig_loader.parser import ZoomHeader
from bigwig_loader.store import BigWigStore
from bigwig_loader.subtract_intervals import subtract_interval_dataframe
from bigwig_loader.util import get_standard_chromosomes


class BigWig:
    def __init__(self, path: Path, id: Optional[int] = None, scale: float = 1.0):
        """
        Create a BigWig object representing one BigWig file.
        Args:
            path: path to BigWig file
            id: integer representing the file in a collection
                when part of a collection.
            scale: scale values in bigwig file by this number.
        """

        with open(path, "rb") as bigwig:
            self.path = path
            self.id = id
            self.scale = scale
            self.bbi_header = BBIHeader.from_file(bigwig)
            self.zoom_headers = ZoomHeader.all(
                bigwig, first_offset=64, n_zoom_levels=self.bbi_header.zoom_levels
            )
            self.total_summary = TotalSummary.from_file_and_offset(
                bigwig, self.bbi_header.total_summary_offset
            )
            self.chromosome_tree_header = ChromosomeTreeHeader.from_file_and_offset(
                bigwig, self.bbi_header.chromosome_tree_offset
            )

            self.chromosome_head_node = ChromosomeTreeNode.from_file_and_offset(
                bigwig,
                key_size=self.chromosome_tree_header.key_size,
                offset=self.bbi_header.chromosome_tree_offset + 32,
            )
            self.chromosome_sizes = {
                chrom.key: chrom.chrom_size for chrom in self.chromosome_head_node.items
            }
            self.rtree_index_header = RTreeIndexHeader.from_file_and_offset(
                bigwig, self.bbi_header.full_index_offset
            )
            self.rtree_head_node = RTreeNode.from_file_and_offset(
                file_object=bigwig,
                start_chrom_ix=self.rtree_index_header.start_chrom_ix,
                start_base=self.rtree_index_header.start_base,
                end_chrom_ix=self.rtree_index_header.end_chrom_ix,
                end_base=self.rtree_index_header.end_base,
                offset=None,
            )

            self.rtree_leaf_nodes = self.rtree_head_node.get_leaf_nodes()
            self.max_rows_per_chunk = self._guess_max_rows_per_chunk(bigwig)

        self.chrom_to_chrom_id: dict[str, int] = {
            item.key: item.chrom_id for item in self.chromosome_head_node.items  # type: ignore
        }
        self._chrom_id_to_chrom = self._create_chrom_id_to_chrom_key()

        self.chromosome_offsets: npt.NDArray[np.int64] = None  # type: ignore
        self.store: BigWigStore = None  # type: ignore
        self.ncls_index: NCLS = None
        self.reference_df: pd.DataFrame = None  # type: ignore

    def run_indexing(self, chromosome_offsets: npt.NDArray[np.int64]) -> None:
        """Run NCLS indexing of BigWig file. The bigwig file has
        an index itself as well. But we prefer to recalculate
        an index here.
        """

        self.chromosome_offsets = chromosome_offsets
        self.ncls_index, self.reference_df = self.build_ncls_index()
        self.store = BigWigStore(
            self.path,
            chunk_sizes=self.reference_df["data_size"].values,  # type: ignore
            chunk_offsets=self.reference_df["data_offset"].values,  # type: ignore
        )

    def chromosomes(
        self,
        include_chromosomes: Union[Literal["all", "standard"], Sequence[str]] = "all",
        exclude_chromosomes: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """
        Get chromosome keys ("chr1", "chr2"...) that are present in
        the BigWig file. Subsets of all chromosome keys found in this file
        can be made by supplying include_chromosomes and exclude_chromosomes.
        Args:
            include_chromosomes: chromosome keys you want to include.
                Can alternatively be "all" or "standard".
            exclude_chromosomes: Optional set of chromosomes to exclude.

        Returns: list of chromosome keys

        """
        exclude_chromosomes = exclude_chromosomes or []
        chromosomes_in_file = list(self.chrom_to_chrom_id.keys())
        if include_chromosomes == "all":
            include_chromosomes = chromosomes_in_file
        elif include_chromosomes == "standard":
            include_chromosomes = [
                chrom
                for chrom in get_standard_chromosomes(exclude=exclude_chromosomes)
                if chrom in chromosomes_in_file
            ]
        elif isinstance(include_chromosomes, str):
            include_chromosomes = [include_chromosomes]
        return [
            chrom for chrom in include_chromosomes if chrom not in exclude_chromosomes
        ]

    def intervals(
        self,
        include_chromosomes: Union[Literal["all", "standard"], Sequence[str]] = "all",
        exclude_chromosomes: Optional[Sequence[str]] = None,
        blacklist: Optional[pd.DataFrame] = None,
        blacklist_buffer: int = 0,
        threshold: Optional[float] = None,
        merge: bool = False,
        merge_allow_gap: int = 0,
        memory_bank: Optional[MemoryBank] = None,
        decoder: Optional[Decoder] = None,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        """
        Get Intervals from the bigwig file.Does not give back data in the
        chromosome order you asked for necessarily.
        Args:
            include_chromosomes: list of chromosome, "standard" or "all" (default).
            exclude_chromosomes: list of chromosomes you want to exclude
            blacklist: pandas dataframe of intervals that you want to
                exclude from the result.
            blacklist_buffer: default 0. Buffer around blacklist intervals to
                exclude.
            threshold: only return intervals of which the value exceeds
                this threshold.
            merge: whether to merge intervals that are directly following
                eachother. The value will be the max value of the merged
                intervals.
            merge_allow_gap: default 0. Allow intervals seperated by size
                merge_allow_gap bases to still be merged.
        Returns: pandas dataframe of intervals (chrom, start, end, value)

        """
        if memory_bank is None:
            memory_bank = MemoryBank(elastic=True)
        if decoder is None:
            decoder = Decoder(
                max_rows_per_chunk=self.max_rows_per_chunk,
                max_uncompressed_chunk_size=self.max_rows_per_chunk * 12 + 24,
                chromosome_offsets=None,
            )

        chromosome_keys = self.chromosomes(
            include_chromosomes=include_chromosomes,
            exclude_chromosomes=exclude_chromosomes,
        )

        # Doing the sort here so that start_chrom_ids are sorted. This prevents
        # a sort down the line when merging intervals. Because of the different
        # order and mapping between chrom_ids and chrom_keys in different bigwig
        # files, it does not work to sort the resulting chunk_ids, as their order
        # is determined by the consensus order that was created over all bigwig
        # files.
        selected_df = self.reference_df[
            self.reference_df["start_chrom_key"].isin(chromosome_keys)
        ].sort_values(by=["start_chrom_ix", "start_base"])

        offsets = selected_df["data_offset"].values
        sizes = selected_df["data_size"].values

        chrom_ids = []
        starts = []
        ends = []
        values = []
        for i in range(0, len(offsets), batch_size):
            memory_bank.reset()
            memory_bank.add_many(
                self.store._fh,
                offsets[i : i + batch_size],
                sizes[i : i + batch_size],
                skip_bytes=2,
            )
            gpu_byte_array, comp_chunks, compressed_chunk_sizes = memory_bank.to_gpu()
            chrom_id, start, end, value, n_rows_for_chunks = decoder.decode(
                gpu_byte_array,
                comp_chunks,
                compressed_chunk_sizes,
            )
            chrom_id = cp.repeat(chrom_id, n_rows_for_chunks.get().tolist())
            value *= self.scale
            if threshold:
                mask = value > threshold
                chrom_id = chrom_id[mask]
                start = start[mask]
                end = end[mask]
                value = value[mask]
            # bringing everything back to CPU
            chrom_ids.append(chrom_id.get())
            starts.append(start.get())
            ends.append(end.get())
            values.append(value.get())
        chrom_ids = np.concatenate(chrom_ids)
        starts = np.concatenate(starts)
        ends = np.concatenate(ends)
        values = np.concatenate(values)

        if merge:
            chrom_ids, starts, ends, values = merge_intervals(
                chrom_ids,
                starts,
                ends,
                values,
                is_sorted=True,
                allow_gap=merge_allow_gap,
            )
        chrom_key = self._chrom_id_to_chrom[chrom_ids]
        data = pd.DataFrame(
            {"chrom": chrom_key, "start": starts, "end": ends, "value": values}
        )
        if blacklist is not None:
            data = subtract_interval_dataframe(
                intervals=data, blacklist=blacklist, buffer=blacklist_buffer
            )

        return data

    def get_batch_offsets_and_sizes_with_global_positions(
        self,
        global_starts: Union[Iterable[int], npt.NDArray[np.int64]],
        global_ends: Union[Iterable[int], npt.NDArray[np.int64]],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Get the offsets and sizes in bytes for the compressed chunks that need
        to be loaded to be able to extract the data belonging to the intervals.

        Args:
            global_starts: start positions that are running over all chromosomes
            global_ends: end positions that are running over all chromosomes

        Returns:
            Tuple of np.ndarray offsets and np.ndarray of sizes in Bytes

        """
        _, right_index = self.search_ncls_with_global_positions(
            global_starts, global_ends
        )
        return self.store.get_offsets_and_sizes(np.sort(np.unique(right_index)))

    def build_ncls_index(self) -> tuple[NCLS, pd.DataFrame]:
        leaf_nodes = self.rtree_leaf_nodes

        df = pd.DataFrame(
            {
                "start_chrom_ix": [
                    leaf_node.start_chrom_ix for leaf_node in leaf_nodes
                ],
                "start_base": [leaf_node.start_base for leaf_node in leaf_nodes],
                "end_chrom_ix": [leaf_node.end_chrom_ix for leaf_node in leaf_nodes],
                "end_base": [leaf_node.end_base for leaf_node in leaf_nodes],
                "data_offset": [leaf_node.data_offset for leaf_node in leaf_nodes],
                "data_size": [leaf_node.data_size for leaf_node in leaf_nodes],
            }
        )
        df["start_chrom_key"] = self._chrom_id_to_chrom[df["start_chrom_ix"].values]  # type: ignore
        df["start_abs"] = self.make_positions_global(
            df["start_chrom_ix"].values, df["start_base"].values  # type: ignore
        )
        df["end_abs"] = self.make_positions_global(
            df["end_chrom_ix"].values, df["end_base"].values  # type: ignore
        )
        df = df.sort_values(by="start_abs").reset_index(drop=True)
        ncls = NCLS(df["start_abs"].values, df["end_abs"].values, df.index.values)
        return ncls, df

    def search_ncls(
        self, query_df: pd.DataFrame, use_key: bool = False
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Use NCLS to search for overlapping intervals. The right key gives the
        chunk id that needs to be loaded from disk to find the interval information
        needed.

        Args:
            query_df: pandas Dataframe with columns "chrom", "start", "end"
            use_key: when use_key is True, the "chrom" column is assumed to
                have chrom keys (i.e. "chr1", "chr2"...). If False, the chrom
                column is assumed to have integer chrom_ids.

        Returns:
            tuple of left indexes and right indexes of overlapping intervals

        """

        start: npt.NDArray[np.int64] = query_df["start"].values  # type: ignore
        end: npt.NDArray[np.int64] = query_df["end"].values  # type: ignore

        if use_key:
            chrom_keys: Iterable[str] = query_df["chrom"].values
            start = np.array(
                self.make_positions_global_with_chromosome_keys(chrom_keys, start)
            )
            end = np.array(
                self.make_positions_global_with_chromosome_keys(chrom_keys, end)
            )
        else:
            chrom_ids: npt.NDArray[np.int64] = query_df["chrom"].values  # type: ignore
            start = np.array(self.make_positions_global(chrom_ids, start))
            end = np.array(self.make_positions_global(chrom_ids, end))
        return self.ncls_index.all_overlaps_both(start, end, query_df.index.values)  # type: ignore

    def search_ncls_with_global_positions(
        self,
        global_starts: Union[Iterable[int], npt.NDArray[np.int64]],
        global_ends: Union[Iterable[int], npt.NDArray[np.int64]],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Use NCLS to search for overlapping intervals. The right key gives the
        chunk id that needs to be loaded from disk to find the interval information
        needed.

        Args:
            global_starts: start positions that are running over all chromosomes
            global_ends: end positions that are running over all chromosomes
        Returns:
            tuple of left indexes and right indexes of overlapping intervals

        """

        index = np.arange(len(global_starts))  # type: ignore
        return self.ncls_index.all_overlaps_both(global_starts, global_ends, index)  # type: ignore

    def make_positions_global_with_chromosome_keys(
        self,
        chromosome_keys: Iterable[str],
        positions: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        chromosomes = [self.chrom_to_chrom_id[key] for key in chromosome_keys]
        return self.make_positions_global(chromosomes, positions)

    def make_positions_global(
        self,
        chromosomes: Union[Sequence[int], npt.NDArray[np.int64]],
        positions: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        return positions + self.chromosome_offsets[chromosomes]

    def _create_chrom_id_to_chrom_key(self) -> npt.NDArray[np.generic]:
        """Create a mapping from chrom_ids (5, 1, 3) to chrom
        keys ("chr1", "chr2"...) in the form of a numpy
        array, for quick conversion.

        Returns:
            numpy array with chrom keys
        """

        chromosomes = [
            (chromosome.chrom_id, chromosome.key)
            for chromosome in self.chromosome_head_node.items
        ]
        largest_id = max(chromosomes)[0]
        mapping = np.empty(largest_id + 1, dtype=object)  # type: ignore
        for chrom_id, key in chromosomes:
            mapping[chrom_id] = key
        return mapping  # type: ignore

    def _guess_max_rows_per_chunk(
        self, file_object: BinaryIO, sample_size: int = 300
    ) -> int:
        """
        Randomly samples some chunks and looks in the header
        to see if in these chunks, the number of rows in the
        decompressed data is always the same. This helps with
        the ReferenceFileSystem.

        Args:
            file_object: BigWig file opened as bytes.
        Returns:
            Number of rows in a chunk
        """

        rows_for_chunks = []
        if len(self.rtree_leaf_nodes) < sample_size:
            sample_leaf_nodes = self.rtree_leaf_nodes
        else:
            sample_leaf_nodes = sample(self.rtree_leaf_nodes, sample_size)
        for leaf_node in sample_leaf_nodes:
            file_object.seek(leaf_node.data_offset, 0)  # type: ignore
            decoded = zlib.decompress(file_object.read(leaf_node.data_size))  # type: ignore
            header = WIGSectionHeader.from_bytes(decoded[:24])
            rows_for_chunks.append(header.item_count)
        return max(rows_for_chunks)
