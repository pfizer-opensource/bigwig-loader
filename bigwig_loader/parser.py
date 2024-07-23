from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO
from typing import Optional

import numpy as np

BBI_HEADER_ENCODING = "<LHHQQQHHQQLQ"
ZOOM_HEADER_ENCODING = "<LLQQ"
TOTAL_SUMMARY_ENCODING = "<Qdddd"
CHROMOSOME_TREE_HEADER_ENCODING = "<LLLLQQ"
CHROMOSOME_TREE_NODE_ENCODING = "<??H"
RTREE_INDEX_HEADER_ENCODING = "<LLQLLLLQLL"
RTREE_NODE_ENCODING = "<??H"
WIG_SECTION_HEADER_ENCODING = "<LLLLLBBH"

NUMPY_RTREE_LEAFNODE_ENCODING = np.dtype(
    [
        ("start_chrom_ix", "u4"),
        ("start_base", "u4"),
        ("end_chrom_ix", "u4"),
        ("end_base", "u4"),
        ("data_offset", "u8"),
        ("data_size", "u8"),
    ]
)

NUMPY_RTREE_NODE_ENCODING = np.dtype(
    [
        ("start_chrom_ix", "u4"),
        ("start_base", "u4"),
        ("end_chrom_ix", "u4"),
        ("end_base", "u4"),
        ("data_offset", "u8"),
    ]
)


@dataclass
class BBIHeader:
    magic: int
    version: int
    zoom_levels: int
    chromosome_tree_offset: int
    full_data_offset: int
    full_index_offset: int
    field_count: int
    defined_field_count: int
    auto_sql_offset: int
    total_summary_offset: int
    uncompress_buff_size: int
    reserved: int

    @classmethod
    def from_file(cls, file_object: BinaryIO) -> BBIHeader:
        return cls.from_bytes(file_object.read(64))

    @classmethod
    def from_bytes(cls, data: bytes) -> BBIHeader:
        return cls(*struct.unpack(BBI_HEADER_ENCODING, data))


@dataclass
class ZoomHeader:
    reduction_level: int
    reserved: int
    data_offset: int
    index_offset: int

    @classmethod
    def all(
        cls, file_object: BinaryIO, first_offset: int, n_zoom_levels: int
    ) -> list[ZoomHeader]:
        zoom_headers = []
        next_level_offset = first_offset
        for level in range(n_zoom_levels):
            zoom_header = cls.from_file_and_offset(file_object, next_level_offset)
            zoom_headers.append(zoom_header)
            next_level_offset = file_object.tell()
        return zoom_headers

    @classmethod
    def from_file_and_offset(cls, file_object: BinaryIO, offset: int) -> ZoomHeader:
        file_object.seek(offset, 0)
        return cls.from_bytes(file_object.read(24))

    @classmethod
    def from_bytes(cls, data: bytes) -> ZoomHeader:
        return cls(*struct.unpack(ZOOM_HEADER_ENCODING, data))


@dataclass
class TotalSummary:
    bases_covered: int
    min_val: float
    max_val: float
    sum_data: float
    sum_squares: float

    @classmethod
    def from_file_and_offset(cls, file_object: BinaryIO, offset: int) -> TotalSummary:
        file_object.seek(offset, 0)
        return cls.from_bytes(file_object.read(40))

    @classmethod
    def from_bytes(cls, data: bytes) -> TotalSummary:
        return cls(*struct.unpack(TOTAL_SUMMARY_ENCODING, data))


@dataclass
class ChromosomeTreeHeader:
    magic: int
    block_size: int
    key_size: int
    val_size: int
    item_count: int
    reserved: int

    @classmethod
    def from_file_and_offset(
        cls, file_object: BinaryIO, offset: int
    ) -> ChromosomeTreeHeader:
        file_object.seek(offset, 0)
        return cls.from_bytes(file_object.read(32))

    @classmethod
    def from_bytes(cls, data: bytes) -> ChromosomeTreeHeader:
        return cls(*struct.unpack(CHROMOSOME_TREE_HEADER_ENCODING, data))


@dataclass
class ChromosomeTreeLeafItem:
    key: str
    chrom_id: Optional[int]
    chrom_size: Optional[int]
    child_offset: Optional[int]
    child_node: Optional[object]


@dataclass
class ChromosomeTreeNode:
    is_leaf: bool
    reserved: bool
    count: int
    items: list[ChromosomeTreeLeafItem]

    @classmethod
    def from_file_and_offset(
        cls, file_object: BinaryIO, key_size: int, offset: Optional[int] = None
    ) -> ChromosomeTreeNode:
        if offset is not None:
            file_object.seek(offset, 0)
        is_leaf, reserved, count = struct.unpack(
            CHROMOSOME_TREE_NODE_ENCODING, file_object.read(4)
        )

        items = []
        location = file_object.tell()
        for i in range(count):
            location = file_object.seek(location, 0)
            chrom_id = None
            chrom_size = None
            child_offset = None
            child_node = None
            bits = file_object.read(key_size + 8)
            if is_leaf:
                key, chrom_id, chrom_size = struct.unpack(f"<{key_size}sLL", bits)
                location = file_object.tell()
            else:
                key, child_offset = struct.unpack(f"<{key_size}Q", bits)
                location = file_object.tell()
                child_node = ChromosomeTreeNode.from_file_and_offset(
                    file_object, key_size=key_size, offset=child_offset
                )

            key = key.decode("utf-8").strip("\x00")
            items.append(
                ChromosomeTreeLeafItem(
                    key, chrom_id, chrom_size, child_offset, child_node=child_node
                )
            )
        return cls(is_leaf, reserved, count, items)


@dataclass
class RTreeIndexHeader:
    magic: int
    block_size: int
    item_count: int
    start_chrom_ix: int
    start_base: int
    end_chrom_ix: int
    end_base: int
    end_file_offset: int
    items_per_slot: int
    reserved: int

    @classmethod
    def from_file_and_offset(
        cls, file_object: BinaryIO, offset: int
    ) -> RTreeIndexHeader:
        file_object.seek(offset, 0)
        return cls.from_bytes(file_object.read(48))

    @classmethod
    def from_bytes(cls, data: bytes) -> RTreeIndexHeader:
        return cls(*struct.unpack(RTREE_INDEX_HEADER_ENCODING, data))


@dataclass
class WIGSectionHeader:
    chrom_id: int
    chrom_start: int
    chrom_end: int
    item_step: int
    item_span: int
    type: str
    item_count: int

    @classmethod
    def from_bytes(cls, data: bytes) -> WIGSectionHeader:
        (
            chrom_id,
            chrom_start,
            chrom_end,
            item_step,
            item_span,
            section_type,
            reserved,
            item_count,
        ) = struct.unpack(WIG_SECTION_HEADER_ENCODING, data)

        section_type = {1: "bed_graph", 2: "variable_step", 3: "fixed_step"}[
            section_type
        ]
        return cls(
            chrom_id,
            chrom_start,
            chrom_end,
            item_step,
            item_span,
            section_type,
            item_count,
        )


@dataclass
class ZoomRecord:
    chrom_id: int
    chrom_start: int
    chrom_end: int
    valid_count: int
    min_val: float
    max_val: float
    sum_data: float
    sum_of_squares: float

    @classmethod
    def from_bytes(cls, data: bytes) -> ZoomRecord:
        return cls(*struct.unpack("<LLLLffff", data))


@dataclass
class ZoomLevel:
    zoom_count: int
    zoom_records: list[ZoomRecord]

    @classmethod
    def from_file_and_offset(
        cls, file_object: BinaryIO, offset: Optional[int] = None
    ) -> ZoomLevel:
        if offset:
            file_object.seek(offset, 0)
        (zoom_count,) = struct.unpack("<L", file_object.read(4))
        print("zoom count:", zoom_count)
        zoom_records = []
        for i in range(zoom_count):
            record = ZoomRecord.from_bytes(file_object.read(32))
            if record.min_val > record.max_val:
                raise ValueError("ZoomRecord min_val is larger than max_val.")
            if record.chrom_start > record.chrom_end:
                raise ValueError("ZoomRecord chrom_start is larger than chrom_end.")
            zoom_records.append(record)
        return cls(zoom_count=zoom_count, zoom_records=zoom_records)


@dataclass
class RTreeNode:
    is_leaf: bool
    children: list[RTreeNode]
    start_chrom_ix: int
    start_base: int
    end_chrom_ix: int
    end_base: int
    data_offset: Optional[int] = None
    data_size: Optional[int] = None

    def get_leaf_nodes(self) -> list[RTreeNode]:
        if self.is_leaf:
            return [self]
        leaf_nodes = []
        for child in self.children:
            leaf_nodes.extend(child.get_leaf_nodes())
        return leaf_nodes

    @classmethod
    def from_file_and_offset(
        cls,
        file_object: BinaryIO,
        start_chrom_ix: int,
        start_base: int,
        end_chrom_ix: int,
        end_base: int,
        offset: Optional[int] = None,
    ) -> RTreeNode:
        current_start_chrom_ix = start_chrom_ix
        current_start_base = start_base
        current_end_chrom_ix = end_chrom_ix
        current_end_base = end_base

        if offset is not None:
            file_object.seek(offset, 0)
        is_leaf, reserved, count = struct.unpack(
            RTREE_NODE_ENCODING, file_object.read(4)
        )

        children = []
        position = file_object.tell()
        if is_leaf:
            for i in range(count):
                (
                    start_chrom_ix,
                    start_base,
                    end_chrom_ix,
                    end_base,
                    data_offset,
                    data_size,
                ) = struct.unpack("<LLLLQQ", file_object.read(32))
                child_node = RTreeNode(
                    is_leaf=True,
                    children=[],
                    start_chrom_ix=start_chrom_ix,
                    start_base=start_base,
                    end_chrom_ix=end_chrom_ix,
                    end_base=end_base,
                    data_offset=data_offset,
                    data_size=data_size,
                )
                children.append(child_node)

        else:
            for i in range(count):
                file_object.seek(position)
                (
                    start_chrom_ix,
                    start_base,
                    end_chrom_ix,
                    end_base,
                    data_offset,
                ) = struct.unpack("<LLLLQ", file_object.read(24))
                position = file_object.tell()

                child_node = RTreeNode.from_file_and_offset(
                    file_object=file_object,
                    start_chrom_ix=start_chrom_ix,
                    start_base=start_base,
                    end_chrom_ix=end_chrom_ix,
                    end_base=end_base,
                    offset=data_offset,
                )
                children.append(child_node)

        return cls(
            is_leaf=False,
            children=children,
            start_chrom_ix=current_start_chrom_ix,
            start_base=current_start_base,
            end_chrom_ix=current_end_chrom_ix,
            end_base=current_end_base,
            data_offset=None,
            data_size=None,
        )


def collect_leaf_nodes(
    file_object: BinaryIO, offset: Optional[int] = None
) -> np.typing.NDArray[np.void]:
    return np.concatenate(_collect_leaf_nodes(file_object, offset))  # type: ignore


def _collect_leaf_nodes(
    file_object: BinaryIO, offset: Optional[int] = None
) -> list[np.typing.NDArray[np.void]]:
    # Assumes the file object is already at the correct position
    if offset is not None:
        file_object.seek(offset, 0)
    is_leaf, reserved, count = struct.unpack(RTREE_NODE_ENCODING, file_object.read(4))

    if is_leaf:
        data = np.fromfile(
            file_object, dtype=NUMPY_RTREE_LEAFNODE_ENCODING, count=count
        )
        return [data]
    else:
        data = np.fromfile(file_object, dtype=NUMPY_RTREE_NODE_ENCODING, count=count)
        data_offsets = data["data_offset"]
        leaf_nodes = []
        for data_offset in data_offsets:
            leaf_node_data = _collect_leaf_nodes(file_object, data_offset)
            leaf_nodes.extend(leaf_node_data)
        return leaf_nodes
