import queue
import threading
from typing import Iterable
from typing import Sequence

import cupy as cp
import numpy as np
import numpy.typing as npt

from bigwig_loader.batch import BatchProcessor
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.intervals_to_values import intervals_to_values


class WorkerContext:
    def __init__(
        self,
        input_queue: queue.Queue[
            tuple[
                Sequence[str] | npt.NDArray[np.generic],
                Sequence[int] | npt.NDArray[np.int64],
                Sequence[int] | npt.NDArray[np.int64],
            ]
        ],
        output_queue: queue.Queue[tuple[cp.ndarray, "WorkerContext"]],
        stop_event: threading.Event,
        collection: BigWigCollection,
        worker_id: int | None = None,
    ):
        self.worker_id = worker_id
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.ready = threading.Event()  # Event to signal when ready for a new batch
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()
        self._collection = collection
        self._batch_processor: BatchProcessor | None = None

    #     )
    @property
    def batch_processor(self) -> BatchProcessor:
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(
                bigwigs=self._collection.bigwigs,
                max_rows_per_chunk=self._collection.max_rows_per_chunk,
                local_to_global=self._collection.make_positions_global,
                local_chrom_ids_to_offset_matrix=self._collection.local_chrom_ids_to_offset_matrix,
                use_cufile=True,
            )
        return self._batch_processor

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            self.ready.wait()  # Wait until the event is set
            self.ready.clear()  # Clear the event to wait for the next signal
            try:
                with self.stream:
                    chrom, start, end = self.input_queue.get(timeout=1)
                    result = self.batch_processor.preprocess(chrom, start, end)
                    self.stream.synchronize()
                    self.output_queue.put((result, self))
                    self.input_queue.task_done()
            except queue.Empty:
                raise StopIteration

    def join(self) -> None:
        self.thread.join()


class StreamedDataloader:
    def __init__(
        self,
        input_generator: Iterable[
            tuple[
                Sequence[str] | npt.NDArray[np.generic],
                Sequence[int] | npt.NDArray[np.int64],
                Sequence[int] | npt.NDArray[np.int64],
            ]
        ],
        collection: BigWigCollection,
        num_threads: int = 4,
        queue_size: int = 10,
    ):
        self.data_generator = input_generator
        self.collection = collection
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.input_queue: queue.Queue[
            tuple[
                Sequence[str] | npt.NDArray[np.generic],
                Sequence[int] | npt.NDArray[np.int64],
                Sequence[int] | npt.NDArray[np.int64],
            ]
        ] = queue.Queue(
            maxsize=self.queue_size
        )  # Thread-safe input queue
        self.output_queue: queue.Queue[
            tuple[cp.ndarray, "WorkerContext"]
        ] = queue.Queue(
            maxsize=self.queue_size
        )  # Thread-safe output queue
        self.stop_event = threading.Event()
        self.workers: list[WorkerContext] = []
        self._create_workers()
        self._out = None

    def _create_workers(self) -> None:
        for i in range(self.num_threads):
            worker = WorkerContext(
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                stop_event=self.stop_event,
                collection=self.collection,
                worker_id=i,
            )
            self.workers.append(worker)

    def __iter__(self) -> "StreamedDataloader":
        self.data_generator_thread = threading.Thread(target=self._feed_generator)
        self.data_generator_thread.start()
        for worker in self.workers:
            worker.ready.set()
        return self

    def _feed_generator(self) -> None:
        for data in self.data_generator:
            self.input_queue.put(data)
            if self.stop_event.is_set():
                break

    def __next__(self) -> cp.ndarray:
        try:
            result, worker = self.output_queue.get(timeout=30)

            (
                start_data,
                end_data,
                value_data,
                abs_start,
                abs_end,
                found_starts,
                found_ends,
            ) = result

            # Perform the last step in the main thread
            final_data = intervals_to_values(
                start_data,
                end_data,
                value_data,
                abs_start,
                abs_end,
                found_starts,
                found_ends,
                out=self._out,
            )
            self.output_queue.task_done()
            # just hang on to this data to
            # reuse
            self._out = final_data
            # Signal the worker that it's ready for a new batch
            worker.ready.set()

            batch = cp.transpose(final_data, (1, 0, 2))
            if self.collection.scaling_factors_cupy is not None:
                batch *= self.collection.scaling_factors_cupy
            return batch

        except queue.Empty:
            raise StopIteration

    def stop(self) -> None:
        self.stop_event.set()
        self.data_generator_thread.join()
        self.join_workers()

    def join_workers(self) -> None:
        for worker in self.workers:
            worker.join()
