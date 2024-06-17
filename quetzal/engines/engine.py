#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import NewType, Union, Any

class AbstractEngine(ABC):
    """
    An abstract base class for defining processing engines to be used within a Pipeline Stage. 
    Engines derived from this class can perform various types of data or file processing.

    Attributes:
        name (str): The name of the engine, used for logging and identification.

    Methods:
        is_video_analyzed: Static method to determine if a video requires further real-time analysis.
        process: Abstract method to process a given file or list of files.
        end: Abstract method called to signal that no more inputs will be processed.
        save_state: Abstract method to save the current state or results of processing.
    """
    name = "Default Name" # name (str): The name of the engine, used for logging and identification.

    @staticmethod
    def is_video_analyzed(video) -> bool:
        """Static method to determine if a video requires further real-time analysis."""

        return False
    
    

    @abstractmethod
    def process(self, file_path: Any) -> Any:
        """Abstract method to process a given file or list of files."""

        pass

    @abstractmethod
    def end(self):
        """Abstract method called to signal that no more inputs will be processed."""
        pass

    @abstractmethod
    def save_state(self, save_path: str):
        """Abstract method to save the current state or results of processing."""
        pass

QueryFrame = NewType("QueryFrame", str)
DataBaseFrame = NewType("DataBaseFrame", str)
FrameMatch = NewType("FrameMatch", list[tuple[QueryFrame, DataBaseFrame]])
WarpedFrame = NewType("WarpedFrame", list[str])
QueryFrame = NewType("QueryFrame", str)
DataBaseFrame = NewType("DataBaseFrame", str)
FrameMatch = NewType("FrameMatch", list[tuple[QueryFrame, DataBaseFrame]])
WarpedFrame = NewType("WarpedFrame", list[str])

class ObjectDetectionEngine(ABC):
    """
    Abstract class for engines performing object detection on images or video frames.

    Attributes:
        name (str): Name of the object detection engine.

    Methods:
        generate_masked_images: Processes an image to identify objects, applying masks and optionally saving the results.
    """
    name = "Default Name"

    def __init__(self, device):
        pass

    @abstractmethod
    def generate_masked_images(
        self, query_image, caption, save_file_path, box_threshold, text_threshold
    ):
        """Processes an image to identify objects, applying masks and optionally saving the results."""
        pass

class AlignmentEngine(ABC):
    """
    Abstract class for engines that perform alignment of video frames, typically used for matching or comparison tasks.

    Attributes:
        name (str): Name of the alignment engine.

    Methods:
        align_frame_list: Aligns frames from two videos, typically a database and a query video, and returns matched frames and any transformations applied.
    """
    name = "Default Name"

    def __init__(self, device):
        pass

    @abstractmethod
    def align_frame_list(
        self, database_video, query_video, overlay
    ) -> tuple[FrameMatch, WarpedFrame]:
        """Aligns frames from two videos, typically a database and a query video, and returns matched frames and any transformations applied."""
        pass
    

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