# Under Development

import asyncio
import logging
import multiprocessing
import os
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from . import engine

class Stage:
    def __init__(self, engine_setup, input_queue: Queue, output_queue: Queue, num_threads=1, save_path=None, verbose=False, total_items=None, stage_num=0):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engines = [engine_setup() for _ in range(num_threads)]
        self.threads = [threading.Thread(target=self.run, args=(engine,)) for engine in self.engines]
        self.save_path = save_path
        self.verbose = verbose
        self.total_items = total_items
        self.pbar = None
        self.stage_num = stage_num
        self.processed = 0
        self.running = False
        self._lock = threading.Lock()


    def start(self):
        self.running = True
        for thread in self.threads:
            thread.start()

    def is_running(self):
        with self._lock:
            return self.running
        
    def end(self):
        self.input_queue.put(None)

    def run(self, engine: engine.AbstractEngine):
        if self.verbose and self.total_items is not None:
            self.pbar = tqdm(total=self.total_items, position=self.stage_num,  desc=f"Stage {self.stage_num + 1}: ({self.engines[0].name})")

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
                    logging.error(f"Unexpected Tailing Input for Stage {self.stage_num}")
                    graceful_end = False
                self.input_queue.task_done()

            ## Save state if the stage is endded correctely
            if graceful_end:
                engine.save_state(self.save_path)

            ## Update running state
            self.running = False

class Pipeline:
    def __init__(self, stages, queue_maxsize=128, verbose=True):
        '''
        stages: (engine_setup, num_threads, total_inputs) tuple
        '''
        self.queues = [Queue(maxsize=queue_maxsize) for _ in range(len(stages) + 1)]
        self.stages = [Stage(engine_setup, self.queues[i], self.queues[i + 1], num_threads, total_items=total_items, verbose=verbose, stage_num=i) 
                       for i, (engine_setup, num_threads, total_items) in enumerate(stages)]
        # self.stages = [Stage(engine_setup, self.queues[i], self.queues[i + 1], num_threads, total=total) 
        #                      for i, (engine_setup, num_threads, total) in enumerate(stages)]

        self.output_list = []
        self._lock = threading.Lock()
        self.executor = None
    
    def start(self):
        self.executor = ThreadPoolExecutor()
        # with ThreadPoolExecutor() as executor:
        for stage in self.stages:
            self.executor.submit(stage.start)

    def submit(self, file_paths):
        '''
        submit new task. Returns number of task submitted
        '''
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
        

    def get_result(self, pbar=None):
        empty = True
        result = None
        with self._lock:
            running_list = [stage for stage in self.stages if stage.is_running()]

            for stage in running_list:
                if not stage.input_queue.empty():
                    empty = False

            # Handle the case where last input is being processed
            if empty and self.stages[-1].is_running():
                self.stages[-1].input_queue.join()
                empty = self.queues[-1].empty()

            # when it is not empty, wait for next result()
            if not empty: 
                result = self.queues[-1].get()
                self.output_list.append(result)
                self.queues[-1].task_done()
                if pbar:
                    pbar.update(1)

            return result
    
    def join_results(self):
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
        with self._lock:
            if self.executor == None:
                logging.error("You must first call start() to run the pipeline before you call end()")
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
