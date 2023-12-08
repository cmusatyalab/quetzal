#!/usr/bin/env python

from abc import ABC, abstractmethod

class AbstractEngine(ABC):

    # @abstractmethod
    # def warmup(self):
    #     '''Initial warm-up if needed
    #     '''
        # pass
    name = "Default Name"

    @abstractmethod
    def process(self, file_path: list):
        '''Process list of files in file_path

        Return an resulting file.'''

        pass

    @abstractmethod
    def end(self):
        '''Indicate no more input will be processed'''
        pass

    @abstractmethod
    def save_state(self, save_path):
        '''Save state in save_path. return None or final result'''
        pass