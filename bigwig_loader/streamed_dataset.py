import queue
import threading
from types import TracebackType
from typing import Generator
from typing import Iterable
from typing import Iterator

import cupy as cp

from bigwig_loader.batch import Batch
from bigwig_loader.batch import IntervalType
from bigwig_loader.batch_processor import BatchProcessor
from bigwig_loader.batch_processor import PreprocessedReturnType
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.intervals_to_values import intervals_to_values

InputBatchType = Batch | IntervalType


class WorkerContext:
    def __init__(
        self,
        input_queue: queue.Queue[InputBatchType],
        output_queue: queue.Queue[tuple[Batch, PreprocessedReturnType, int]],
        stop_event: threading.Event,
        collection: BigWigCollection,
        worker_id: int,
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
        while self._wait_until_ready():
            try:
                # necessary because some other thread
                # (for instance pytorch-lightning)
                # can call cuda.set_current_device triggering
                # a device mismatch
                with cp.cuda.Device(self.stream.device_id):
                    with self.stream as stream:
                        query = self.input_queue.get(timeout=1)
                        query = Batch.from_args(query)
                        result = self.batch_processor.preprocess(
                            chromosomes=query.chromosomes,
                            start=query.starts,
                            end=query.ends,
                            track_indices=query.track_indices,
                            stream=stream,
                        )
                        self.stream.synchronize()
                        self._put_output_queue(query, result)
                        self.input_queue.task_done()
            except queue.Empty:
                self.ready.set()
                continue
            self._batch_count += 1

    def _wait_until_ready(self) -> bool:
        while True:
            if self.stop_event.is_set():
                return False
            # returns True when ready event is set
            # and false when the timeout is reached
            if self.ready.wait(timeout=1):
                self.ready.clear()
                return True
        return True

    def _put_output_queue(self, query: Batch, result: PreprocessedReturnType) -> None:
        while not self.stop_event.is_set():
            try:
                self.output_queue.put((query, result, self.worker_id), timeout=1)
                break
            except queue.Full:
                continue

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
        window_size: int = 1,
    ):
        self.input_generator = input_generator
        self.collection = collection
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.slice_size = slice_size
        self.window_size = window_size
        self.input_queue: queue.Queue[InputBatchType] = queue.Queue(
            maxsize=self.queue_size
        )  # Thread-safe input queue
        self.output_queue: queue.Queue[tuple[Batch, cp.ndarray, int]] = queue.Queue(
            maxsize=self.queue_size
        )  # Thread-safe output queue
        self.stop_event = threading.Event()
        self.workers: list[WorkerContext] = []
        self.main_stream = cp.cuda.Stream(non_blocking=True)
        self.data_generator_thread: threading.Thread | None = None
        self._entered = False
        self._out = None

    def __enter__(self) -> "StreamedDataloader":
        self._entered = True
        self.stop_event.clear()
        self._create_workers()
        self.data_generator_thread = threading.Thread(target=self._feed_generator)
        self.data_generator_thread.start()
        for worker in self.workers:
            worker.ready.set()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._entered = False
        self._destroy()

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

    def __iter__(self) -> Iterator[Batch]:
        if not self._entered:
            raise RuntimeError(
                "StreamedDataloader must be used within a 'with' statement"
            )
        return self._generate_batches()

    def _feed_generator(self) -> None:
        for data in self.input_generator:
            while not self.stop_event.is_set():
                try:
                    self.input_queue.put(data, timeout=1)
                    break
                except queue.Full:
                    continue
            if self.stop_event.is_set():
                break

    def _generate_batches(self) -> Generator[Batch, None, None]:
        try:
            while True:
                batch, result, worker_id = self.output_queue.get(timeout=30)

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

                        values = cp.transpose(value_matrix, (1, 0, 2))
                        if self.collection.scaling_factors_cupy is not None:
                            scaling_factors = self.collection.scaling_factors_cupy
                            if batch.track_indices is not None:
                                scaling_factors = scaling_factors[
                                    :, batch.track_indices, :
                                ]

                            values *= scaling_factors
                        stream.synchronize()

                        sliced_query = batch[select]
                        sliced_query.values = values

                    yield sliced_query

                self.output_queue.task_done()
                self.workers[worker_id].ready.set()
        except queue.Empty:
            self.stop()
            return
        except Exception as e:
            self.stop()
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
        self._join_data_generator()
        self._join_workers()

    def _join_workers(self) -> None:
        for worker in self.workers:
            worker.join()

    def _join_data_generator(self) -> None:
        if (
            self.data_generator_thread is not None
            and self.data_generator_thread.is_alive()
        ):
            self.data_generator_thread.join()

    def _destroy(self) -> None:
        self.stop()
        self._destroy_workers()
        self._empty_queue()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    def _destroy_workers(self) -> None:
        self.workers = []

    def _empty_queue(self) -> None:
        while not self.input_queue.empty():
            self.input_queue.get()
            self.input_queue.task_done()
        while not self.output_queue.empty():
            self.output_queue.get()
            self.output_queue.task_done()

    def __del__(self) -> None:
        self._destroy()
