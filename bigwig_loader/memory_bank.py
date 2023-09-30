import logging
from typing import BinaryIO
from typing import Iterable
from typing import Optional

import cupy as cp
import numpy as np


class MemoryBank:
    def __init__(self, nbytes: int = 10000, elastic: bool = True):
        self._mem = cp.cuda.alloc_pinned_memory(nbytes)
        self._view = memoryview(self._mem)
        # self._cpu_array =
        self._current_file_handle: Optional[BinaryIO] = None
        self.n_chunks = 0
        self.compressed_chunk_sizes: list[int] = []
        self.compressed_chunk_offsets: list[int] = [0]
        self.cumulative_n_chunks_per_file: list[int] = []
        self.elastic = elastic

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

    def increase_memory_size(self, new_size: int, generosity: float = 1.2) -> None:
        """
        Increase the size of the pinned memory when needed.
        Args:
            new_size: new size of pinned memory needed
            generosity: when increasing the memory, increase more
                by this factor to prevent having to update too often.

        """
        if new_size > self._mem.size():
            new_size = int(new_size * generosity)
            logging.debug(f"Increasing size of pinned memory to {new_size}")
            new_mem = cp.cuda.alloc_pinned_memory(new_size)
            new_view = memoryview(new_mem)
            new_view[: self._view.nbytes] = self._view
            self._mem = new_mem
            self._view = new_view

    def add(
        self, file_handle: BinaryIO, offset: int, size: int, skip_bytes: int = 2
    ) -> None:
        """
        Add one compressed chunk to the pinned memory.
        Args:
            file_handle: opened file
            offset: byte offset to start of compressed chunk
            size: byte size of compressed chunk
            skip_bytes: skips this number of bytes from offset
            increase_memory_size: whether to elastically increase the
                size of the pinned memory if the new chunk doesn't
                fit in anymore. (default: True)
        Returns:

        """
        offset += skip_bytes
        size -= skip_bytes
        if file_handle is not self._current_file_handle:
            self.cumulative_n_chunks_per_file.append(self.n_chunks)
            self._current_file_handle = file_handle
        self.n_chunks += 1
        current_offset = self.compressed_chunk_offsets[-1]
        new_offset = current_offset + size
        file_handle.seek(offset)
        # file_handle.readinto(self._view[current_offset:new_offset])
        if self.elastic:
            self.increase_memory_size(new_offset)
        file_handle.readinto(self._view[current_offset:new_offset])  # type: ignore
        self.compressed_chunk_offsets.append(new_offset)
        self.compressed_chunk_sizes.append(size)

    def add_many(
        self,
        file_handle: BinaryIO,
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

    def to_gpu(self) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Brings the data from pinned memory to GPU.
        Returns: everything needed for decompression on GPU:
            Tuple(gpu byte array, pointers to starts of compressed
                  chunks, sizes of compressed chunks)
        """
        compressed_chunk_sizes = cp.array(self.compressed_chunk_sizes, dtype=cp.uint64)
        # offsets = cp.array(self.compressed_chunk_offsets[:-1], dtype=cp.uint64)
        offsets = cp.pad(cp.cumsum(compressed_chunk_sizes), (1, 0))[:-1]

        cpu_array = np.frombuffer(
            self._mem, dtype="byte", count=self.compressed_chunk_offsets[-1]
        )

        logging.debug(f"CPU Array {cpu_array}")

        # gpu_byte_array = cp.asarray(self._mem)[:self.compressed_chunk_offsets[-1]]
        gpu_byte_array = cp.asarray(cpu_array)
        comp_chunks = offsets + gpu_byte_array.data.ptr
        return gpu_byte_array, comp_chunks, compressed_chunk_sizes


if __name__ == "__main__":
    bank = MemoryBank()
    file_handle = open("look.txt", "rb", buffering=0)

    bank.add(file_handle, offset=0, size=16)
    bank.add(file_handle, offset=16, size=16)

    file_handle = open("new.prof", "rb", buffering=0)

    bank.add(file_handle, offset=0, size=16)
    bank.add(file_handle, offset=16, size=16)

    print(np.frombuffer(bank._mem, dtype="int", count=100))
    print(bank.to_gpu())

    print(bank.compressed_chunk_offsets)
    print(bank.cumulative_n_chunks_per_file)

    bank.reset()

    print("reset----")

    file_handle = open("look.txt", "rb", buffering=0)

    bank.add(file_handle, offset=0, size=16)
    bank.add(file_handle, offset=16, size=16)

    file_handle = open("new.prof", "rb", buffering=0)

    bank.add(file_handle, offset=0, size=16)
    bank.add(file_handle, offset=16, size=16)

    print(np.frombuffer(bank._mem, dtype="byte", count=100))
    print("to gpu:")
    print(bank.to_gpu())

    print(bank.compressed_chunk_offsets)
    print(bank.cumulative_n_chunks_per_file)
