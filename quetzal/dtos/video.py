import logging
import os
import re
import threading
from glob import glob
from os.path import join
from typing import Literal, List
from typing import NewType, Union
from pathlib import Path
import importlib

import cv2
import numpy as np
import natsort
import shutil

from quetzal.engines.engine import AbstractEngine
from quetzal.dtos.dtos import QuetzalFile, Permission, FileType, AnalysisProgress, AccessMode
from quetzal.utils.video_tools import extract_frames
import torch

logging.basicConfig()


def convert_path(original_path, resolution):
    path_obj = Path(original_path)
    
    parts = list(path_obj.parts)
    parent_dir = parts[-2]
        
    # Check if the parent directory follows the 'frame_{}_{}' convention
    if parent_dir.startswith('frames_') and parent_dir.count('_') == 2:
        base, fps, _ = parent_dir.split('_')
        new_parent_dir = f"{base}_{fps}_{resolution}"
        
        parts[-2] = new_parent_dir
        new_path = Path(*parts)
        return str(new_path)
    else:
        print("The parent directory does not follow the expected 'frames_{}_{}' convention.")
        return None


def extract_fps_res(directory_name):
    """
    Extracts frames per second (fps) and resolution from a directory name.

    Args:
        directory_name (str): The directory name from which to extract the fps and resolution.

    Returns:
        tuple: A tuple (fps, resolution) if extraction is successful, otherwise (None, None).
    """

    # Regular expression to match 'frames_{fps}_{resolution}'
    pattern = re.compile(r"frames_(\d+)_(\d+)")
    match = pattern.search(directory_name)
    if match:
        fps = int(match.group(1))
        resolution = int(match.group(2))
        return fps, resolution
    else:
        print(f"Invalid directory name: {directory_name}")
        return None, None


def frame_count(fp):
    """
    Reads the number of frames from a file.

    Args:
        fp (str): File path containing the frame count.

    Returns:
        int: The number of frames.
    """
    return int(open(fp, "r").read().strip())



Resolution = NewType("Resolution", int)
Fps = NewType("Fps", int)

DATABASE_ROOT = "database"
QUERY_ROOT = "query"
reserved_names = [DATABASE_ROOT, QUERY_ROOT]
# VideoTypes: TypeAlias = Literal[*reserved_names]

class Video(QuetzalFile):
    
    """
    A class to manage video processing for drone footage.

    Attributes:
        root_dir (str, Path): The root directory for datasets.
        metadata_dir (str, Path): root directory for metadata.
        path (str, Path): Path to the video file, relative to root_dir.
        video_type (Literal["database", "query"]): The type of the video, either 'database' or 'query'.
        fps (int): The frames per second at which to process the video.
        resolution (int): The resolution for the processed frames.
        ...
    """
    
    FILE_DEFAULT_DESCRIPTION = "Uploader::= default\nRecorded Date (MM/DD/YYYY)::= default\nTime-of-day::= default\nWeather Condition::= default\nDescription::= default"
    FILE_DEFAULT_META = "FileType::= file\nVisibility::= private\nPermission::= full_write\nAnalysisProgress::= none\nSpecificType::=Video\n"
        
    def __init__(
        self,
        path: Union[str, Path],
        root_dir: Union[str, Path],
        metadata_dir: Union[str, Path],        
        user: str = None,
        home: Union[str, Path] = "./",
        metadata = None,
        parent: 'QuetzalFile' = None,
        video_type: Literal["database", "query"] = "query",
        fps: Fps = 2,
        resolution: Resolution = 1024,
    ):
        super().__init__(
            path=path,
            root_dir=root_dir,
            metadata_dir=metadata_dir,
            user=user,
            home=home,
            metadata=metadata,
            parent=parent
        )
                
        assert video_type in [
            "database",
            "query",
        ], "video_type must be 'database' or 'query'"
        
        self._video_type = video_type
        self._fps = fps
        self._resolution = resolution
        
        self._gps = None
        self.logger = logging.getLogger("Video_" + self._path.name)
        self.logger.setLevel(logging.DEBUG)
        self.debug = lambda *args: self.logger.debug(" ".join([str(arg) for arg in args]))
        
        self._lock = threading.RLock()
        self._cam = None
        self._native_fps = None
        self._orig_width, self._orig_height = None, None
        
        self._video_start = 0
        self._video_end = None
        
    
    def load_video_info(self):
        # Initialize the camera and check if it's a valid video file
        self._cam = cv2.VideoCapture(str(self.full_path))
        if not self._cam.isOpened():
            raise ValueError(f"Failed to open video file: {self.full_path}")

        self._native_fps = self._cam.get(cv2.CAP_PROP_FPS)
        self._orig_width, self._orig_height = int(self._cam.get(3)), int(self._cam.get(4))
        
    @property
    def fps(self) -> Fps:
        return self._fps
    
    @fps.setter
    def fps(self, value: Fps):
        self._fps = value
    
    @property
    def resolution(self) -> Resolution:
        return self._resolution
    
    @resolution.setter
    def resolution(self, value: Resolution):
        self._resolution = value
    
    @property
    def video_type(self) -> Literal["database", "query"] :
        return self._video_type
    
    @video_type.setter
    def video_type(self, value: Literal["database", "query"] ):
        self._video_type = value
        
    @property
    def _dataset_dir(self) -> Path:
        if self._metadata_dir:
            return self._metadata_dir / self._path.parent / self._video_type / self._path.stem
        else:
            return self._root_dir / self._path.parent / self._video_type / self._path.stem
        
    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir / f"frames_{self._fps}_{self._resolution}"
    
    
    @property
    def avaliable_frames(self) -> list[tuple[Fps, Resolution]]:
        """
        Checks the availability of different combinations of fps and resolution for which the frames are extracted.

        Returns:
            list[tuple]: A list of tuples (fps, resolution) available in the dataset.
        """
        dir_list = [
            d
            for d in os.listdir(self._dataset_dir)
            if os.path.isdir(join(self._dataset_dir, d))
        ]

        if len(dir_list) == 0:
            return []

        return [
            extract_fps_res(str(dir))
            for dir in dir_list
            if self.is_dir_valid(join(self._dataset_dir, dir))
        ]
    
    
    @property
    def frame_len(self) -> int:
        """
        Retrieves the total number of frames in the dataset directory.

        Returns:
            int: The number of frames, or -1 if the frame count is not available.
        """

        target_dir = self.dataset_dir
        frame_count_file = join(target_dir, "frame_counts.txt")

        with self._lock:
            if os.path.exists(frame_count_file):
                return frame_count(frame_count_file)
            else:
                return -1
    

    @staticmethod
    def get_frame_list(dir: str) -> List[str]:
        """
        Get a list of frame file paths from a directory.

        Args:
            dir (str): The directory to search for frame files.

        Returns:
            List[str]: A list of file paths for each frame in the directory.
        """
        files = glob(join(dir, f"frame*.jpg"))
        pattern = re.compile(r"frame(\d{5})\.jpg")
        return [f for f in files if pattern.search(f)]

    def get_frame_idx(self, frame_path: str) -> int:
        """
        Extracts the frame index from a frame file path.

        Args:
            frame_path (str): The file path of the frame.

        Returns:
            int: The index of the frame.
        """

        frame_number_str = frame_path[-9:-4]  # Assuming 5-digit frame number
        return int(frame_number_str) - 1

    def get_frame_idx_at_time(self, frame_time: float) -> int:
        """
        Gets the frame index at a specific time in the video.

        Args:
            frame_time (float): The time in seconds.

        Returns:
            int: The frame index corresponding to the given time.
        """
        return int(frame_time * self._fps + 0.5)

    def get_frame_time(self, frame_index: int) -> float:
        """
        Converts a frame index to a time point in the video.

        Args:
            frame_index (int): The index of the frame.

        Returns:
            float: Time in seconds where the frame is located in the video.
        """
        return frame_index / self._fps

    def get_raw_frame_at(self, idx: int) -> np.ndarray:
        """
        Retrieves the raw frame at a given index from the video.

        Args:
            idx (int): The frame index.

        Returns:
            ndarray: The raw frame as a numpy array in RGB format, or None if retrieval fails.
        """
        if self._cam is None:
            self.load_video_info()

        orig_frame_idx = int(idx / self._fps * self._native_fps + 0.5)
        self._cam.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
        rv, frame = self._cam.read()

        if rv:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    def set_video_start(self, minute: int, sec: int):
        """
        Sets the start point of the video for processing.

        Args:
            minute (int): The start minute in the video.
            sec (int): The start second in the video.
        """
        

        if not (minute >= 0 and sec >= 0):
            logging.error("Video-start should be positive time")
            return

        sec = 60 * minute + sec
        index = sec * self._fps
        self._video_start = index

    def set_video_end(self, minute: int, sec: int):
        """
        Sets the end point of the video for processing.

        Args:
            minute (int): The end minute in the video.
            sec (int): The end second in the video.
        """

        if not (minute >= 0 and sec >= 0):
            logging.error("Video-end should be positive time")
            return

        sec = 60 * minute + sec
        index = sec * self._fps
        if index > self._video_start:
            self._video_end = index
        else:
            minute = int(self._video_start / self._fps / 60)
            sec = int(self._video_start / self._fps) % 60
            logging.error(
                f"Video-end Should be greator than video-start = {minute}:{sec}"
            )
            self._video_end = None

    def get_video_start_idx(self) -> int:
        """
        Gets the starting frame index of the video for processing.

        Returns:
            int: The starting frame index.
        """
        return self._video_start

    def get_video_end_idx(self) -> int:
        """
        Gets the ending frame index of the video for processing.

        Returns:
            int: The ending frame index, or None if not set.
        """
        return self._video_end


    def is_dir_valid(self, dir: Union[str, Path]) -> bool:
        """
        Checks if a directory is a valid dataset directory with extracted frames

        Args:
            dir (str): The directory to check.

        Returns:
            bool: True if the directory is valid, False otherwise.
        """

        if not os.path.isdir(dir):
            return False

        frame_count_file = join(dir, "frame_counts.txt")
        if os.path.exists(frame_count_file):
            if len(Video.get_frame_list(dir)) == frame_count(frame_count_file):
                return True
        return False

    def get_frames(self, verbose: bool = True) -> List[str]:
        """
        Retrieves or generates frames for the video at the current fps and resolution settings.

        Args:
            verbose (bool, optional): If True, logs additional information. Defaults to True.

        Returns:
            list: A list of frame file paths.
        """

        fps = self._fps
        resolution = self._resolution

        target_dir = self.dataset_dir
        frame_count_file = target_dir / "frame_counts.txt"

        loaded = False
        with self._lock:
            if not self.is_dir_valid(target_dir):
                if verbose:
                    self.logger.info(
                        f"Frames in fps={fps}, res={resolution} do not exist. Generating..."
                    )
                os.makedirs(target_dir, exist_ok=True)
                input_frames = extract_frames(
                    self.full_path, target_dir, max_size=resolution, fps=fps
                )
                with open(frame_count_file, "w") as f:
                    f.write(str(len(input_frames)))

                loaded = True

        if not loaded:
            if verbose:
                self.logger.info(
                    f"Frames in fps={fps}, res={resolution} found in the dataset."
                )
            input_frames = Video.get_frame_list(target_dir)

        # self.frame_len = len(input_frames)
        idx_start = self._video_start
        if self._video_end == None:
            idx_end = len(input_frames)
        else:
            idx_end = self._video_end
        return natsort.natsorted(input_frames)[
            idx_start : min(idx_end, len(input_frames))
        ]    
        
    def _renameAnalysisData(self, newName: Path):
        for data_dir in reserved_names:
            analysis_path = (
                self._metadata_dir / self._path.parent / data_dir / self._path.stem
            )

            if analysis_path.exists():
                new_path = (
                    self._metadata_dir / self._path.parent / data_dir / newName.stem
                )
                os.rename(analysis_path, new_path)
    
    def _updateMetaForRename(self, new_path):
        self._renameAnalysisData(new_path)
        super()._updateMetaForRename(new_path)
    
    
    def _deleteAnalysisData(self):
        """Delete associated database and query analysis data."""
        for data_dir in [DATABASE_ROOT, QUERY_ROOT]:
            analysis_path = (
                self._metadata_dir / self._path.parent / data_dir / self._path.stem
            )

            if analysis_path.exists():
                shutil.rmtree(analysis_path)
                
    def _updateMetaForDelete(self):
        self._deleteAnalysisData()
        super()._updateMetaForDelete()
        
    
    def _copyAnalysisData(self, newDir: Path, destName: Path, move=False):
        for data_dir in ["database", "query"]:
            analysis_path = (
                self._metadata_dir / self._path.parent / data_dir / self._path.stem
            )
            if analysis_path.exists():
                copy_path = self._metadata_dir / newDir / data_dir / destName.stem
                if move:
                    shutil.move(analysis_path, copy_path)
                else:
                    shutil.copytree(analysis_path, copy_path)
    
    def _updateMetaForCopy(self, dest: Path):
        super()._updateMetaForCopy(dest)
        self._copyAnalysisData(dest.parent, Path(dest.name))
        
    
    def _updateMetaForMove(self, dest_dir: Path):
        super()._updateMetaForMove(dest_dir)
        self._copyAnalysisData(dest_dir, Path(self._path.name), move=True)


    def _syncAnalysisState(self, engine="vpr_engine.anyloc_engine.AnyLocEngine"):
        assert self._type == FileType.FILE
        # from quetzal.engines.vpr_engine.anyloc_engine import AnyLocEngine
        
        module_path, class_name = engine.rsplit(".", 1)
        if module_path:
            module = importlib.import_module(f"quetzal.engines.{module_path}")
            engine_class: AbstractEngine = getattr(module, class_name)
        
        if engine_class is None:
            raise ValueError("Engine Definition Not Found")
        
        video = DatabaseVideo(
            path=self._path,
            root_dir=self._root_dir,
            metadata_dir=self._metadata_dir,
            user=self._user,
            home=self._home,
        )
            
        if engine_class.is_video_analyzed(video):
            self._updateMetaForAnalyze(new_progress=AnalysisProgress.FULL)
            self._analysis_progress = AnalysisProgress.FULL
            return

        video = QueryVideo(
            path=self._path,
            root_dir=self._root_dir,
            metadata_dir=self._metadata_dir,
            user=self._user,
            home=self._home,
        )

        if engine_class.is_video_analyzed(video):
            self._updateMetaForAnalyze(new_progress=AnalysisProgress.HALF)
            self._analysis_progress = AnalysisProgress.HALF
            return

        self._updateMetaForAnalyze(new_progress=AnalysisProgress.NONE)
        self._analysis_progress = AnalysisProgress.NONE
        return


    def _analyze(self, option: AnalysisProgress, engine="vpr_engine.anyloc_engine.AnyLocEngine", device=torch.device("cuda:0")):
        assert (
            self._mode == AccessMode.OWNER or self._permission != Permission.READ_ONLY
        )
        self.debug(f"\n\t{self.name} called on analyze {option}\n")
        
        if option == None:
            return None

        self._syncAnalysisState(engine)
        if self._analysis_progress >= option:
            return None
        
        module_path, class_name = engine.rsplit(".", 1)
        if module_path:
            module = importlib.import_module(f"quetzal.engines.{module_path}")
            engine_class: AbstractEngine = getattr(module, class_name)
        
        if engine_class is None:
            raise ValueError("Engine Definition Not Found")
        
        self.debug(engine_class)
                
        if option == AnalysisProgress.FULL:
            video = DatabaseVideo(
                path=self._path,
                root_dir=self._root_dir,
                metadata_dir=self._metadata_dir,
                user=self._user,
                home=self._home,
            )
        if option == AnalysisProgress.HALF:
            video = QueryVideo(
                path=self._path,
                root_dir=self._root_dir,
                metadata_dir=self._metadata_dir,
                user=self._user,
                home=self._home,
            )
            
        engine_class(device = device).analyze_video(video)

        self._updateMetaForAnalyze(new_progress=option)
        self._analysis_progress = option
        
        return f'"{self.name}" Analysis Done'

    
class DatabaseVideo(Video):
    """
    A subclass of Video for processing database-type videos.

    Inherits all attributes and methods from Video.
    """
    
    FPS = 6
    RESOLUTION = 1024

    def __init__(
        self, 
        path: Union[str, Path],
        root_dir: Union[str, Path],
        metadata_dir: Union[str, Path],        
        user: str = None,
        home: Union[str, Path] = "./", 
        metadata = None,
        parent: 'QuetzalFile' = None,
    ):
        """
        Initialize the DatabaseVideo object with dataset directory, project, and video name.

        Args:
            datasets_dir (str): The root directory for datasets.
            project_name (str): The name of the specific project.
            video_name (str): The name of the video file.

        Dataset structure
            root_datsets_dir/
            |
            ├── project_name/
            |   ├── raw_video/
            |   |   ├── video_name.mp4
            |   |   └── ...
            |   |
            |   ├── database/
            |   |   ├── video_name/
            |   |   |   ├── frames_{fps}_{resolution}/
            |   |   |   |   ├── frame_%05.jpg
            |   |   |   |   └── ...
            |   |   |   └── ...
            |   |   └── ...
            |   |
            |   ├── query/
            |   |   ├── video_name/
            |   |   |   ├── frames_{fps}_{resolution}/
            |   |   |   |   ├── frame_%05.jpg
            |   |   |   |   └── ...
            |   |   |   └── ...
            |   |   └── ...
            |   └── ...
            └── ...
        """

        # Override video_type to be "database"
        super().__init__(
            path=path,
            root_dir=root_dir,
            metadata_dir=metadata_dir,
            user=user,
            metadata=metadata,
            parent=parent,
            video_type="database",
            home=home,
            fps=self.FPS,
            resolution=self.RESOLUTION,
        )
    
    @staticmethod
    def from_quetzal_file(file: QuetzalFile):
        database_video = DatabaseVideo(
            path=file._path,
            root_dir=file._root_dir,
            metadata_dir=file._metadata_dir,
            user=file._user,
            home=file._home,
        )
        
        return database_video


class QueryVideo(Video):
    """
    A subclass of Video for processing query-type videos.

    Inherits all attributes and methods from Video.
    """
    FPS = 2
    RESOLUTION = 1024
    
    def __init__(
        self, 
        path: Union[str, Path],
        root_dir: Union[str, Path],
        metadata_dir: Union[str, Path],        
        user: str = None,
        home: Union[str, Path] = "./", 
        metadata = None,
        parent = None,
    ):
        """
        Initialize the QueryVideo object with dataset directory, project, and video name.

        Args:
            datasets_dir (str): The root directory for datasets.
            project_name (str): The name of the specific project.
            video_name (str): The name of the video file.

        dataset_dir: root datasets dir
        project_name: name of the specific project the drone video is taken
        video_name: video file name

        Dataset structure
            root_datsets_dir/
            |
            ├── project_name/
            |   ├── raw_video/
            |   |   ├── video_name.mp4
            |   |   └── ...
            |   |
            |   ├── database/
            |   |   ├── video_name/
            |   |   |   ├── frames_{fps}_{resolution}/
            |   |   |   |   ├── frame_%05.jpg
            |   |   |   |   └── ...
            |   |   |   └── ...
            |   |   └── ...
            |   |
            |   ├── query/
            |   |   ├── video_name/
            |   |   |   ├── frames_{fps}_{resolution}/
            |   |   |   |   ├── frame_%05.jpg
            |   |   |   |   └── ...
            |   |   |   └── ...
            |   |   └── ...
            |   └── ...
            └── ...
        """

        # Override video_type to be "database"
        super().__init__(
            path=path,
            root_dir=root_dir,
            metadata_dir=metadata_dir,
            user=user,
            home=home,
            metadata=metadata,
            parent=parent,
            video_type="query",
            fps=self.FPS,
            resolution=self.RESOLUTION,
        )
        
    def _convert_from_database(self, verbose):
        database_video = DatabaseVideo(
            path=self._path,
            root_dir=self._root_dir,
            metadata_dir=self._metadata_dir,
            user=self._user   
        )
        database_target = database_video.dataset_dir
        if not database_video.is_dir_valid(database_target):
            return False
        
        if verbose:
            self.logger.info(
                f"Extracing Frames in fps={self.fps}, res={self.resolution} from Database..."
            )
            
        target_dir = self.dataset_dir
        os.makedirs(target_dir, exist_ok=True)
        
        database_frames = database_video.get_frames(verbose=False)
        database_frames = [os.path.basename(frame) for frame in database_frames]
        
        every_third_frame = database_frames[::3]
        for i, frame in enumerate(every_third_frame):
            old_path = os.path.join(database_target, frame)
            new_frame_name = f"frame{i+1:05d}.jpg"
            new_path = os.path.join(target_dir, new_frame_name)
            
            # Copying the file
            shutil.copy2(old_path, new_path)
            
        frame_count_file = join(target_dir, "frame_counts.txt")
        with open(frame_count_file, "w") as f:
            f.write(str(len(every_third_frame)))
        
        return True

        
    def get_frames(self, verbose: bool = True) -> List[str]:
        """
        Retrieves or generates frames for the video at the current fps and resolution settings.

        Args:
            verbose (bool, optional): If True, logs additional information. Defaults to True.

        Returns:
            list: A list of frame file paths.
        """

        target_dir = self.dataset_dir

        with self._lock:
            if not self.is_dir_valid(target_dir):
                os.makedirs(target_dir, exist_ok=True)           
                self._convert_from_database(verbose)               

            frame_list = super().get_frames()
            
        return frame_list

    @staticmethod
    def from_quetzal_file(file: QuetzalFile):
        query = QueryVideo(
            path=file._path,
            root_dir=file._root_dir,
            metadata_dir=file._metadata_dir,
            user=file._user,
            home=file._home,
        )
        return query
        


# TODO
class LiveQueryVideo(Video):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, video_type="query", **kwargs)

    def get_frames(self, fps=2, resolution=1024):
        super().get_frames(fps, resolution)



