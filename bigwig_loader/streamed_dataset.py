import queue
import threading
from typing import Generator
from typing import Iterable
from typing import Sequence

import cupy as cp
import numpy as np
import numpy.typing as npt

from bigwig_loader.batch import BatchProcessor
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.intervals_to_values import intervals_to_values

InputBatchType = tuple[
    Sequence[str] | npt.NDArray[np.generic],
    Sequence[int] | npt.NDArray[np.int64],
    Sequence[int] | npt.NDArray[np.int64],
]


class WorkerContext:
    def __init__(
        self,
        input_queue: queue.Queue[InputBatchType],
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
        self._batch_count: int = 0

    @property
    def batch_processor(self) -> BatchProcessor:
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(
                bigwigs=self._collection.bigwigs,
                max_rows_per_chunk=self._collection.max_rows_per_chunk,
                local_to_global=self._collection.make_positions_global,
                local_chrom_ids_to_offset_matrix=self._collection.local_chrom_ids_to_offset_matrix,
            )
        return self._batch_processor

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            self.ready.wait()  # Wait until the event is set
            self.ready.clear()  # Clear the event to wait for the next signal
            try:
                with self.stream as stream:
                    chrom, start, end = self.input_queue.get(timeout=1)
                    result = self.batch_processor.preprocess(
                        chrom, start, end, stream=stream
                    )
                    self.stream.synchronize()
                    self.output_queue.put((result, self))
                    self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f" worker {self.worker_id}; batch {self._batch_count}")
                print(
                    f" worker {self.worker_id}; batch {self._batch_count}",
                    "chrom",
                    chrom,
                )
                print(
                    f" worker {self.worker_id}; batch {self._batch_count}",
                    "start",
                    start,
                )
                print(
                    f" worker {self.worker_id}; batch {self._batch_count}", "end", end
                )
                raise e
            self._batch_count += 1

    def join(self) -> None:
        self.stream.synchronize()
        self.thread.join()


class StreamedDataloader:
    def __init__(
        self,
        input_generator: Iterable[InputBatchType],
        collection: BigWigCollection,
        num_threads: int = 4,
        queue_size: int = 10,
        slice_size: int | None = None,
        window__size: int = 1,
    ):
        self.input_generator = input_generator
        self.collection = collection
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.slice_size = slice_size
        self.window_size = window__size
        self.input_queue: queue.Queue[InputBatchType] = queue.Queue(
            maxsize=self.queue_size
        )  # Thread-safe input queue
        self.output_queue: queue.Queue[tuple[cp.ndarray, WorkerContext]] = queue.Queue(
            maxsize=self.queue_size
        )  # Thread-safe output queue
        self.stop_event = threading.Event()
        self.workers: list[WorkerContext] = []
        self.main_stream = cp.cuda.Stream(non_blocking=True)
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

    def __iter__(self) -> Iterable[cp.ndarray]:
        self.data_generator_thread = threading.Thread(target=self._feed_generator)
        self.data_generator_thread.start()
        for worker in self.workers:
            worker.ready.set()
        return self._generate_batches()

    def _feed_generator(self) -> None:
        for data in self.input_generator:
            self.input_queue.put(data)
            if self.stop_event.is_set():
                break

    def _generate_batches(self) -> Generator[cp.ndarray, None, None]:
        try:
            while True:
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

                n_samples = len(abs_start)
                slice_size = self._determine_slice_size(n_samples=n_samples)

                out = self._get_out_tensor(
                    sequence_length=(abs_end[0] - abs_start[0]).item()
                    // self.window_size,
                    number_of_tracks=found_starts.shape[0],
                    batch_size=slice_size,
                )

                for select in self._slices_objects(n_samples, slice_size):
                    with self.main_stream as stream:
                        stream.synchronize()

                        value_matrix = intervals_to_values(
                            array_start=start_data,
                            array_end=end_data,
                            array_value=value_data,
                            query_starts=abs_start[select],
                            query_ends=abs_end[select],
                            found_starts=found_starts[:, select],
                            found_ends=found_ends[:, select],
                            window_size=self.window_size,
                            out=out,
                        )

                        batch = cp.transpose(value_matrix, (1, 0, 2))
                        if self.collection.scaling_factors_cupy is not None:
                            batch *= self.collection.scaling_factors_cupy
                        stream.synchronize()

                    yield value_matrix

                self.output_queue.task_done()
                worker.ready.set()
        except queue.Empty:
            self.stop_event.set()
            return
        except Exception as e:
            print(f"Main thread encountered an error: {e}")
            self.stop_event.set()
            raise e

    def _get_out_tensor(
        self, number_of_tracks: int, batch_size: int, sequence_length: int
    ) -> cp.ndarray:
        shape = (number_of_tracks, batch_size, sequence_length)
        if self._out is None or self._out.shape != shape:
            self._out = cp.zeros(shape, dtype=cp.float32)
        return self._out

    def _determine_slice_size(self, n_samples: int) -> int:
        if self.slice_size is None:
            return n_samples
        if self.slice_size > n_samples:
            return n_samples
        return self.slice_size

    def _slices_objects(
        self, n_samples: int, slice_size: int
    ) -> Generator[slice, None, None]:
        for i in range(n_samples // slice_size):
            yield slice(i * slice_size, (i + 1) * slice_size)

    def stop(self) -> None:
        self.stop_event.set()
        self.data_generator_thread.join()
        self.join_workers()

    def join_workers(self) -> None:
        for worker in self.workers:
            worker.join()

    def __del__(self) -> None:
        self.stop()
