# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import math
import random
import time
from multiprocessing import Process, Queue
from typing import List, Callable

import mxnet as mx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class AsyncDataLoader:
    def __init__(self, datapoints: List[str], split_and_batchify: Callable, n_batch: int, ctx: List[mx.context.Context],
                 n_workers: int):
        self.datapoints = datapoints
        self.split_and_batchify = split_and_batchify
        self.n_batch = n_batch
        self.ctx = ctx
        self.n_workers = n_workers
        self.total_batches = int(math.ceil(len(datapoints) / n_batch))

    def __enter__(self):
        self.task_queue = Queue()
        self.results_queue = Queue(maxsize=self.n_workers)
        self.worker_processes = []
        for _ in range(self.n_workers):
            time.sleep(1)
            p = Process(target=self.worker_process,
                        args=(self.task_queue,
                              self.results_queue,
                              self.split_and_batchify,
                              [mx.cpu(i) for i in range(len(self.ctx))]
                              # Have to process on cpu in worker, load to gpu in __next__
                              ),
                        daemon=True)
            p.start()
            self.worker_processes.append(p)

        self.batches_released = 0
        random.shuffle(self.datapoints)
        for b in range(self.total_batches):
            self.task_queue.put(self.datapoints[self.n_batch * b: self.n_batch * (b + 1)])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _ in self.worker_processes:
            self.task_queue.put('STOP')
        for p in self.worker_processes:
            p.terminate()
        del self.task_queue
        del self.results_queue
        del self.worker_processes

    def __len__(self):
        return len(self.datapoints)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_released >= self.total_batches:
            raise StopIteration
        else:
            start_time = time.time()
            split_batch, batch_length = self.results_queue.get()
            queue_time = time.time()
            for batch, ctx in zip(split_batch, self.ctx):
                batch.move_to_context(ctx)
            self.batches_released += 1
            return_time = time.time()
            logger.debug('Time to get batch off queue: {}'.format(queue_time - start_time))
            logger.debug('Time to return batch: {}'.format(return_time - start_time))
            return split_batch, batch_length

    @staticmethod
    def worker_process(task_queue: Queue, results_queue: Queue, split_and_batchify: Callable,
                       ctx: List[mx.context.Context]):
        for data_filepaths in iter(task_queue.get, 'STOP'):
            split_batch, batch_length = split_and_batchify(data_filepaths, ctx)
            results_queue.put((split_batch, batch_length))
