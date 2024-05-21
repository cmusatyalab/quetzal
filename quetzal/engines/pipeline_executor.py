# Under Development

import asyncio
import logging
import multiprocessing
import os
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Callable, Optional, Union, NewType, TypeAlias, Any
from quetzal.engines.engine import AbstractEngine

from . import engine

StageInput = NewType("StageInput", Any)
StageOuput = NewType("StageOuput", Any)

class Stage:
    """
    Represents a stage in a pipeline processing flow, managing the execution of tasks across multiple threads.
    """
    
    def __init__(
        self,
        engine_setup: Callable[[], AbstractEngine],
        input_queue: Queue,
        output_queue: Queue,
        num_threads: int=1,
        save_path: Optional[str]=None,
        verbose: bool=False,
        total_items: Optional[int]=None,
        stage_num: int=0,
    ):
        """
        Attributes:
            engine_setup (callable): A function that returns an instance of the engine to be used for processing.
            input_queue (Queue): The queue from which input tasks are retrieved.
            output_queue (Queue): The queue where processed tasks are placed.
            num_threads (int): The number of threads allocated for task processing.
            save_path (str, optional): Path to save any necessary data or results.
            verbose (bool): Enables detailed logging if set to True.
            total_items (int, optional): The total number of items expected to be processed. Useful for progress tracking.
            stage_num (int): Identifier for the stage, typically used for logging and progress tracking.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engines = [engine_setup() for _ in range(num_threads)]
        self.threads = [
            threading.Thread(target=self.run, args=(engine,)) for engine in self.engines
        ]
        self.save_path = save_path
        self.verbose = verbose
        self.total_items = total_items
        self.pbar = None
        self.stage_num = stage_num
        self.processed = 0
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        """
        Initializes and starts the threads for processing.
        """
        self.running = True
        for thread in self.threads:
            thread.start()

    def is_running(self) -> bool:
        """
        Checks if the stage is currently processing tasks.
        """
        with self._lock:
            return self.running

    def end(self):
        """
        Signals the end of input, allowing threads to terminate gracefully.
        """
        self.input_queue.put(None)

    def run(self, engine: engine.AbstractEngine):
        """
        The main method executed by each thread, processing tasks from the input queue.
        """
        if self.verbose and self.total_items is not None:
            self.pbar = tqdm(
                total=self.total_items,
                position=self.stage_num,
                desc=f"Stage {self.stage_num + 1}: ({self.engines[0].name})",
            )

        while True:
            file_path = self.input_queue.get()

            if file_path is None:  # Sentinel value to stop processin
                break

            result = engine.process(file_path)
            if result is not None:
                self.output_queue.put(result)

            self.processed += 1
            if self.pbar:
                self.pbar.update(1)

            # for tasks that total_items are known, end the thread after processing all the tasks
            if self.total_items is not None and (self.processed >= self.total_items):
                break

            self.input_queue.task_done()

        with self._lock:
            if self.pbar:
                self.pbar.close()

            ## indicate end to the engine
            rv = engine.end()
            self.output_queue.put(rv)
            self.input_queue.task_done()

            ## Done by processing all items
            graceful_end = True
            while not self.input_queue.empty():
                rv = self.input_queue.get()
                if rv != None:
                    logging.error(
                        f"Unexpected Tailing Input for Stage {self.stage_num}"
                    )
                    graceful_end = False
                self.input_queue.task_done()

            ## Save state if the stage is endded correctely
            if graceful_end:
                engine.save_state(self.save_path)

            ## Update running state
            self.running = False


class Pipeline:
    """
    Orchestrates a sequence of processing stages, encapsulating them within a 
    pipeline architecture powered by thread pools. This setup enables efficient 
    task flow management through queues, facilitating a structured approach to 
    complex data processing operations.

    Workflow Overview:
    Submissions enter through the `submit` method and are placed in the input 
    queue. These submissions then traverse through each stage of the pipeline. 
    Each stage operates in separate threads, consuming outputs from its 
    predecessor and supplying its processed results to the next stage. 
    The final outputs are collected in the Output Queue, accessible via the 
    `get_result` method.

    Pipeline Structure:
    - Input (via `submit` method)
    - Input Queue
    - Stage #1 processing
    - Stage #2 processing
    - ...
    - Output Queue
    - Results retrieval (via `get_result` method)

    Each stage runs in parallel threads, allowing for concurrent processing 
    and efficient throughput from input to final results.
    """
    
    def __init__(
        self, 
        stages: list[tuple], 
        queue_maxsize: int=128, 
        verbose: bool=True
    ):
        """
        Attributes:
            stages (list of tuples): Configuration for each stage, including the engine setup, number of threads, and total inputs.
            queue_maxsize (int): The maximum size for the queues between stages, controlling flow and backpressure.
            verbose (bool): If set to True, enables detailed logging across all stages.
        """
        self.queues = [Queue(maxsize=queue_maxsize) for _ in range(len(stages) + 1)]
        self.stages = [
            Stage(
                engine_setup,
                self.queues[i],
                self.queues[i + 1],
                num_threads,
                total_items=total_items,
                verbose=verbose,
                stage_num=i,
            )
            for i, (engine_setup, num_threads, total_items) in enumerate(stages)
        ]

        self.output_list = []
        self._lock = threading.Lock()
        self.executor = None

    def start(self):
        """
        Initializes and starts all stages in the pipeline.
        """
        self.executor = ThreadPoolExecutor()
        # with ThreadPoolExecutor() as executor:
        for stage in self.stages:
            self.executor.submit(stage.start)

    def submit(self, file_paths: StageInput) -> int:
        """
        Submits a file or a list of files for processing through the pipeline.
        """
        with self._lock:
            if not self.stages[0].is_running():
                logging.error("The pipeline stages are not running")
                return 0

            if isinstance(file_paths, (list, tuple)):
                for file_path in file_paths:
                    self.queues[0].put(file_path)
                return len(file_paths)
            else:
                self.queues[0].put(file_paths)
                return 1

    def get_result(self, pbar: tqdm=None) -> StageOuput:
        """Retrieves the next available result from the final output queue."""
        empty = True
        result = None
        with self._lock:
            running_list = [stage for stage in self.stages if stage.is_running()]

            empty = all(stage.input_queue.empty() for stage in running_list)

            # Handle the case where last input is being processed
            if empty and self.stages[-1].is_running():
                self.stages[-1].input_queue.join()
                empty = self.queues[-1].empty()

            # when it is not empty, wait for next result()
            if not empty:
                result = self.queues[-1].get(block=True,timeout=30)
                self.output_list.append(result)
                self.queues[-1].task_done()
                if pbar:
                    pbar.update(1)

            return result

    def join_results(self):
        """Waits for all submitted tasks to complete and collects all results."""
        with self._lock:
            running_list = [stage for stage in self.stages if stage.is_running()]
            for stage in running_list:
                stage.input_queue.join()

            while not self.queues[-1].empty():
                item = self.queues[-1].get()
                self.output_list.append(item)
                self.queues[-1].task_done()

            return self.output_list

    def end(self):
        """Gracefully terminates all stages and threads in the pipeline."""
        with self._lock:
            if self.executor == None:
                logging.error(
                    "You must first call start() to run the pipeline before you call end()"
                )
                return -1

            running_list = [stage for stage in self.stages if stage.is_running()]
            for stage in running_list:
                stage.input_queue.join()

            running_list = [stage for stage in self.stages if stage.is_running()]
            for stage in running_list:
                stage.end()

            for stage in running_list:
                stage.input_queue.join()

            self.executor.shutdown(wait=True)
            self.executor = None

            for q in self.queues:
                with q.mutex:
                    q.queue.clear()

            return 0

        # for stage in self.stages:
        #     stage.input_queue.join()

        # for stage in self.stages:
        #     stage.input_queue.put(None)

        #     # for _ in range(len(stage.threads)):
        #     #     stage.input_queue.put(None)

        # for stage in self.stages:
        #     stage.input_queue.join()


## LET pdoc3 to generate documentation for private methods 
__pdoc__ = {name: True
            for name, klass in globals().items()
            if name.startswith('_') and isinstance(klass, type)}
__pdoc__.update({f'{name}.{member}': True
                 for name, klass in globals().items()
                 if isinstance(klass, type)
                 for member in klass.__dict__.keys()
                 if member not in {'__module__', '__dict__', 
                                   '__weakref__', '__doc__'}})