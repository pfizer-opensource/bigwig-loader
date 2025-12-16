import logging
from pathlib import Path
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

from bigwig_loader.decompressor import Decoder
from bigwig_loader.memory_bank import MemoryBank
from bigwig_loader.merge_intervals import merge_intervals
from bigwig_loader.parser import BBIHeader
from bigwig_loader.parser import ChromosomeTreeHeader
from bigwig_loader.parser import ChromosomeTreeNode
from bigwig_loader.parser import RTreeIndexHeader
from bigwig_loader.parser import TotalSummary
from bigwig_loader.parser import ZoomHeader
from bigwig_loader.parser import collect_leaf_nodes
from bigwig_loader.store import BigWigStore
from bigwig_loader.subtract_intervals import subtract_interval_dataframe
from bigwig_loader.util import get_standard_chromosomes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BigWig:
    def __init__(
        self,
        path: Path,
        id: Optional[int] = None,
        scale: float = 1.0,
    ):
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

            # Endianness check using BBI header magic
            # UCSC BigWig spec: magic is 0x888FFC26
            # Parser uses little-endian unpacking, so valid BigWig files will read as 0x888FFC26
            # (the bytes in file are 26 FC 8F 88, which when read little-endian gives 0x888FFC26)
            magic = self.bbi_header.magic
            if magic == 0x888FFC26:
                logger.debug(f"BBI header magic: {magic:#x} (valid BigWig file)")
            elif magic == 0x26FC8F88:
                # This would mean the file has bytes 88 8F FC 26, which is byte-swapped
                logger.warning(
                    f"BBI header magic: {magic:#x} (byte-swapped BigWig - may need big-endian parsing)"
                )
            else:
                logger.warning(
                    f"BBI header magic: {magic:#x} (unexpected value - not a valid BigWig file?)"
                )

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

            self.rtree_leaf_nodes = collect_leaf_nodes(file_object=bigwig, offset=None)
            self.max_rows_per_chunk = self.get_max_rows_per_chunk()

        self.chrom_to_chrom_id: dict[str, int] = {
            item.key: item.chrom_id for item in self.chromosome_head_node.items  # type: ignore
        }
        self._chrom_id_to_chrom = self._create_chrom_id_to_chrom_key()

        self.chromosome_offsets: npt.NDArray[np.int64] = None  # type: ignore
        self.store: BigWigStore = None  # type: ignore
        self.ncls_index: NCLS = None
        self.reference_data: npt.NDArray[np.void] = None  # type: ignore

    def run_indexing(self, chromosome_offsets: npt.NDArray[np.int64]) -> None:
        """Run NCLS indexing of BigWig file. The bigwig file has
        an index itself as well. But we prefer to recalculate
        an index here.
        """
        self.chromosome_offsets = chromosome_offsets
        sorted_data = prepare_index_for_bigwig(
            chromosome_offsets=chromosome_offsets,
            rtree_leaf_nodes=self.rtree_leaf_nodes,
        )
        self.finalize_indexing(sorted_data)

    def finalize_indexing(self, sorted_data: npt.NDArray[np.void]) -> None:
        """
        Finalize indexing by building NCLS from prepared data.
        This should be called in the main thread/process.
        """
        self.reference_data = sorted_data
        self.ncls_index = NCLS(
            sorted_data["start_abs"],
            sorted_data["end_abs"],
            np.arange(len(sorted_data)),
        )
        self.store = BigWigStore(
            self.path,
            chunk_sizes=sorted_data["data_size"],
            chunk_offsets=sorted_data["data_offset"],
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
        allowed_chrom_ids = np.array(
            [self.chrom_to_chrom_id[key] for key in chromosome_keys]
        )
        mask = np.isin(self.reference_data["start_chrom_ix"], allowed_chrom_ids)
        filtered_array = self.reference_data[mask]
        sort_indices = np.lexsort(
            (filtered_array["start_base"], filtered_array["start_chrom_ix"])
        )
        sorted_array = filtered_array[sort_indices]

        offsets = sorted_array["data_offset"]
        sizes = sorted_array["data_size"]

        chrom_ids = []
        starts = []
        ends = []
        values = []
        for i in range(0, len(offsets), batch_size):
            memory_bank.reset()
            memory_bank.add_many(
                file_handle=self.store.file_handle,
                offsets=offsets[i : i + batch_size],
                sizes=sizes[i : i + batch_size],
                skip_bytes=2,
            )
            comp_chunk_pointers, compressed_chunk_sizes = memory_bank.to_gpu()
            chrom_id, start, end, value, n_rows_for_chunks = decoder.decode(
                comp_chunk_pointers,
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
        return positions + self.chromosome_offsets[chromosomes]  # type: ignore

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

    def get_max_rows_per_chunk(self) -> int:
        """
        Determines the maximum number of rows in any chunk.

        Uses the BBI header's uncompress_buff_size as the authoritative source,
        since this is the buffer size the file was created with and guarantees
        all chunks can be decompressed into this space.

        Args:
            file_object: BigWig file opened as bytes.
            sample_size: Number of chunks to sample for verification.
        Returns:
            Maximum number of rows that could fit in the uncompressed buffer.
        """
        # The BBI header's uncompress_buff_size is the authoritative value
        # It defines the maximum uncompressed size for any chunk in the file
        # Each row is 12 bytes, plus 24-byte header
        return (self.bbi_header.uncompress_buff_size - 24) // 12


def prepare_index_for_bigwig(
    chromosome_offsets: npt.NDArray[np.int64],
    rtree_leaf_nodes: npt.NDArray[np.void],
) -> npt.NDArray[np.generic]:
    """
    Standalone function for preparing index data in a separate process.
    This can be pickled and run with ProcessPoolExecutor.

    Args:
        path: Path to bigwig file (for identification)
        bigwig_id: ID of the bigwig file
        chromosome_offsets: Offsets for converting local to global positions
        rtree_leaf_nodes: The rtree leaf nodes from the bigwig file

    Returns:
        sorted data
    """
    dtype = np.dtype(
        [
            ("start_chrom_ix", "u4"),
            ("start_base", "u4"),
            ("end_chrom_ix", "u4"),
            ("end_base", "u4"),
            ("data_offset", "u8"),
            ("data_size", "u8"),
            ("start_abs", "i8"),
            ("end_abs", "i8"),
        ]
    )

    data = np.empty(rtree_leaf_nodes.shape, dtype=dtype)
    data["start_chrom_ix"] = rtree_leaf_nodes["start_chrom_ix"]
    data["start_base"] = rtree_leaf_nodes["start_base"]
    data["end_chrom_ix"] = rtree_leaf_nodes["end_chrom_ix"]
    data["end_base"] = rtree_leaf_nodes["end_base"]
    data["data_offset"] = rtree_leaf_nodes["data_offset"]
    data["data_size"] = rtree_leaf_nodes["data_size"]

    # make_positions_global inline
    data["start_abs"] = (
        rtree_leaf_nodes["start_base"]
        + chromosome_offsets[rtree_leaf_nodes["start_chrom_ix"]]
    ).astype(np.int64)
    data["end_abs"] = (
        rtree_leaf_nodes["end_base"]
        + chromosome_offsets[rtree_leaf_nodes["end_chrom_ix"]]
    ).astype(np.int64)

    sort_indices = np.argsort(data["start_abs"])
    sorted_data = data[sort_indices]

    return sorted_data
