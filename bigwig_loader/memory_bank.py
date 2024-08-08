import logging
from typing import Iterable

import cupy as cp
from kvikio.cufile import CuFile
from kvikio.cufile import IOFuture


class MemoryBank:
    def __init__(self, nbytes: int = 10000, elastic: bool = True):
        self._default_nbytes = nbytes
        self._gpu_byte_array = cp.empty(nbytes, dtype=cp.uint8)
        self._current_file_handle: CuFile = None
        self.n_chunks = 0
        self.compressed_chunk_sizes: list[int] = []
        self.compressed_chunk_offsets: list[int] = [0]
        self.cumulative_n_chunks_per_file: list[int] = []
        self.elastic = elastic
        self._promises: list[IOFuture] = []

    def reset(self) -> None:
        """
        Reset all the containing data to empty values. Can be used
        before a new batch comes in.
        """
        self._current_file_handle = None
        self.n_chunks = 0
        self.compressed_chunk_sizes = []
        self.compressed_chunk_offsets = [0]
        self.cumulative_n_chunks_per_file = []
        self._promises = []
        self._gpu_byte_array *= 0

    def shrink(self) -> None:
        """
        Shrink the pinned memory to the minimum size.
        """
        self._gpu_byte_array = cp.empty(self._default_nbytes, dtype=cp.uint8)

    def await_all_promises(self) -> list[int]:
        """
        Waits for all the pending read promises to finish.
        Returns: list of sizes of the succesfully read chunks
        """
        results = [promise.get() for promise in self._promises]
        self._promises = []
        return results

    def increase_memory_size(self, new_size: int, generosity: float = 1.2) -> None:
        """
        Increase the size of the pinned memory when needed.
        Args:
            new_size: new size of pinned memory needed
            generosity: when increasing the memory, increase more
                by this factor to prevent having to update too often.

        """
        if new_size > self._gpu_byte_array.size:
            self.await_all_promises()
            new_size = int(new_size * generosity)
            logging.debug(
                f"CuFileMemoryBank: Increasing size of cupy byte from {self._gpu_byte_array.size} to {new_size}"
            )
            new_mem = cp.empty(new_size, dtype=cp.uint8)
            new_mem[: self._gpu_byte_array.size] = self._gpu_byte_array
            self._gpu_byte_array = new_mem

    def add(
        self, file_handle: CuFile, offset: int, size: int, skip_bytes: int = 2
    ) -> None:
        """
        Add one compressed chunk to the pinned memory.
        Args:
            file_handle: opened file
            offset: byte offset to start of compressed chunk
            size: byte size of compressed chunk
            skip_bytes: skips this number of bytes from offset
        Returns:

        """
        # cast this to python int because when it's uint64 and we add/subtract
        # skip_bytes, it somehow becomes float64, which we don't want.
        offset = int(offset)
        size = int(size)
        offset += skip_bytes
        size -= skip_bytes
        if file_handle is not self._current_file_handle:
            self.cumulative_n_chunks_per_file.append(self.n_chunks)
            self._current_file_handle = file_handle
        self.n_chunks += 1

        current_offset = self.compressed_chunk_offsets[-1]
        new_offset = current_offset + size

        if self.elastic:
            self.increase_memory_size(new_offset)
        promise = file_handle.pread(
            buf=self._gpu_byte_array[current_offset:new_offset],
            size=size,
            file_offset=offset,
        )
        self._promises.append(promise)
        self.compressed_chunk_offsets.append(new_offset)
        self.compressed_chunk_sizes.append(size)

    def add_many(
        self,
        file_handle: CuFile,
        offsets: Iterable[int],
        sizes: Iterable[int],
        skip_bytes: int = 2,
    ) -> None:
        """
        Add many compressed chunks from a single file.
        Args:
            file_handle: opened file
            offsets: byte offsets to starts of compressed chunks
            sizes: sizes of compressed chunks
            skip_bytes: skips this number of bytes from offset
            increase_memory_size: whether to elastically increase the
                size of the pinned memory if the new chunks don't
                fit in anymore. (default: True)

        """
        new_size = self.compressed_chunk_offsets[-1] + sum(sizes)
        if self.elastic:
            # better increase once for an entire batch than separately for each chunk
            self.increase_memory_size(new_size)
        for offset, size in zip(offsets, sizes):
            self.add(
                file_handle,
                offset,
                size,
                skip_bytes=skip_bytes,
            )

    def to_gpu(self) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Brings the data from pinned memory to GPU.
        Returns: everything needed for decompression on GPU:
            Tuple(gpu byte array, pointers to starts of compressed
                  chunks, sizes of compressed chunks)
        """
        compressed_chunk_sizes = cp.array(self.compressed_chunk_sizes, dtype=cp.uint64)
        # offsets = cp.array(self.compressed_chunk_offsets[:-1], dtype=cp.uint64)
        offsets = cp.pad(cp.cumsum(compressed_chunk_sizes), (1, 0))[:-1]
        self.await_all_promises()
        comp_chunk_pointers = offsets + self._gpu_byte_array.data.ptr
        return comp_chunk_pointers, compressed_chunk_sizes
