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

Resolution = NewType("Resolution", int)
Fps = NewType("Fps", int)

DATABASE_ROOT = "database"
QUERY_ROOT = "query"
reserved_names = [DATABASE_ROOT, QUERY_ROOT]

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


def extract_fps_res(directory_name: str) -> Union[tuple[Fps, Resolution], tuple[None, None]]:
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

def frame_count(fp: str) -> str:
    """
    Reads the number of frames from a file.

    Args:
        fp (str): File path containing the frame count.

    Returns:
        int: The number of frames.
    """
    return int(open(fp, "r").read().strip())


# VideoTypes: TypeAlias = Literal[*reserved_names]

class Video(QuetzalFile):
    """
    Represents a video file within the Quetzal system, capable of processing drone footage.

    Attributes:
        video_type (Literal["database", "query"]): Specifies the type of the video, indicating
            its role within the dataset as either a reference (database) or subject (query) video.
        fps (Fps): Frames per second at which the video is processed.
        resolution (Resolution): Resolution at which the video frames are processed.
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
        """
        Initializes a Video object.

        Args:
            path: The path to the video file, relative to `root_dir`.
            root_dir: The root directory for datasets.
            metadata_dir: The root directory for metadata.
            user: The user associated with the video file. Defaults to a guest user.
            home: The base home directory path. Defaults to "./".
            metadata: Metadata associated with the video file.
            parent: The parent QuetzalFile object, if applicable.
            video_type: The type of the video, either 'database' or 'query'. Defaults to 'query'.
            fps: The frames per second at which the video is processed. Defaults to 2.
            resolution: The resolution at which the video frames are processed. Defaults to 1024.
        """
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
        """
        Loads video information such as native frames per second (fps), original width, and height from the video file. Initializes the video capture object.

        Raises:
            ValueError: If the video file cannot be opened.
        """
        # Initialize the camera and check if it's a valid video file
        self._cam = cv2.VideoCapture(str(self.full_path))
        if not self._cam.isOpened():
            raise ValueError(f"Failed to open video file: {self.full_path}")

        self._native_fps = self._cam.get(cv2.CAP_PROP_FPS)
        self._orig_width, self._orig_height = int(self._cam.get(3)), int(self._cam.get(4))
        
    @property
    def fps(self) -> Fps:
        """
        Gets the frames per second (fps) at which the video is processed.

        Returns:
            Fps: The current fps setting for video processing.
        """
        return self._fps
    
    @fps.setter
    def fps(self, value: Fps):
        """
        Sets the frames per second (fps) for video processing.

        Args:
            value (Fps): The new fps setting.
        """
        self._fps = value
    
    @property
    def resolution(self) -> Resolution:
        """
        Gets the resolution at which the video frames are processed.

        Returns:
            Resolution: The current resolution setting for video processing.
        """
        return self._resolution
    
    @resolution.setter
    def resolution(self, value: Resolution):
        """
        Sets the resolution for video processing.

        Args:
            value (Resolution): The new resolution setting.
        """
        self._resolution = value
    
    @property
    def video_type(self) -> Literal["database", "query"] :
        """
        Gets the type of the video, indicating whether it's used as reference (database) or subject (query).

        Returns:
            Literal["database", "query"]: The current video type setting.
        """
        return self._video_type
    
    @video_type.setter
    def video_type(self, value: Literal["database", "query"] ):
        """
        Sets the type of the video to either 'database' or 'query'.

        Args:
            value (Literal["database", "query"]): The new video type setting.
        """
        self._video_type = value
        
    @property
    def _dataset_dir(self) -> Path:
        """
        Internal method to construct the dataset directory path based on the video type and file path.

        Returns:
            Path: The dataset directory path.
        """
        if self._metadata_dir:
            return self._metadata_dir / self._path.parent / self._video_type / self._path.stem
        else:
            return self._root_dir / self._path.parent / self._video_type / self._path.stem
        
    @property
    def dataset_dir(self) -> Path:
        """
        Constructs the complete dataset directory path including frames configuration (fps and resolution).

        Returns:
            Path: The complete dataset directory path for storing frames.
        """
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
        Sets the starting point for video processing based on a specific time given in minutes and seconds.

        Args:
            minute (int): The starting minute in the video.
            sec (int): The starting second in the video.
        """
        

        if not (minute >= 0 and sec >= 0):
            logging.error("Video-start should be positive time")
            return

        sec = 60 * minute + sec
        index = sec * self._fps
        self._video_start = index

    def set_video_end(self, minute: int, sec: int):
        """
        Sets the ending point for video processing based on a specific time given in minutes and seconds.

        Args:
            minute (int): The ending minute in the video.
            sec (int): The ending second in the video.
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
        Validates if the specified directory contains a valid set of extracted frames and matches the recorded frame count.

        Args:
            dir (Union[str, Path]): The directory to validate.

        Returns:
            bool: True if the directory contains a valid set of frames, False otherwise.
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
        """
        Rename analysis data associated with the video, including updating paths of analysis data directories to reflect the new video name.

        Args:
            newName (Path): The new name for the video, used to update analysis data directories.
        """
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
        """
        Overrides the superclass method to update metadata and analysis data for the video in response to renaming. Ensures consistency between the video file's metadata and its analysis data following a name change.

        Args:
            new_path (Path): The new path for the video file, reflecting its new name.
        """
        self._renameAnalysisData(new_path)
        super()._updateMetaForRename(new_path)
        super()._updateMetaForRename(new_path)
    
    
    def _deleteAnalysisData(self):
        """
        Delete all analysis data associated with the video. Ensures that orphaned analysis data is removed when a video file is deleted from the system.
        """
        for data_dir in [DATABASE_ROOT, QUERY_ROOT]:
            analysis_path = (
                self._metadata_dir / self._path.parent / data_dir / self._path.stem
            )

            if analysis_path.exists():
                shutil.rmtree(analysis_path)
                
    def _updateMetaForDelete(self):
        """
        Overrides the superclass method to update the video's metadata to reflect its deletion, including removing all references to the video and its analysis data.
        """
        self._deleteAnalysisData()
        super()._updateMetaForDelete()
        
    
    def _copyAnalysisData(self, newDir: Path, destName: Path, move=False):
        """
        Copy or move the analysis data associated with the video. Supports video copy or move operations by maintaining the integrity of analysis data.

        Args:
            newDir (Path): Directory to which the analysis data will be copied or moved.
            destName (Path): Name of the destination directory or file.
            move (bool, optional): If True, analysis data is moved; otherwise, it is copied. Defaults to False.
        """
        
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
        """
        Overrides the superclass method to update metadata for a video after copying. Includes copying the analysis data to the new location and updating the video's metadata.

        Args:
            dest (Path): Destination path where the video and its analysis data have been copied.
        """
        super()._updateMetaForCopy(dest)
        self._copyAnalysisData(dest.parent, Path(dest.name))
        
    
    def _updateMetaForMove(self, dest_dir: Path):
        """
        Overrides the superclass method to update metadata for a video after moving. Manages relocation of analysis data and updates the video's metadata to reflect the move.

        Args:
            dest_dir (Path): Destination directory to which the video and its analysis data have been moved.
        """
        super()._updateMetaForMove(dest_dir)
        self._copyAnalysisData(dest_dir, Path(self._path.name), move=True)


    def _syncAnalysisState(self, engine="vpr_engine.anyloc_engine.AnyLocEngine"):
        """
        Overrides to synchronize the video's analysis state with a specified analysis engine. Updates the video's metadata to reflect current analysis state.

        Args:
            engine (str, optional): Qualified class name of the analysis engine for state checking. Defaults to "vpr_engine.anyloc_engine.AnyLocEngine".

        Raises:
            ValueError: If the analysis engine cannot be found or initialized.
        """
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
        """
        Overrides to analyze the video based on a specified level of analysis progress, using an analysis engine and device. Updates the video's analysis state and metadata upon completion.

        Args:
            option (AnalysisProgress): Desired analysis level.
            engine (str, optional): Qualified class name of the analysis engine. Defaults to "vpr_engine.anyloc_engine.AnyLocEngine".
            device (torch.device, optional): Computing device for analysis. Defaults to "cuda:0".

        Returns:
            str: Message indicating analysis completion.
        """
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
        """
        Creates a DatabaseVideo instance from an existing QuetzalFile object. This method allows for the easy conversion of a general file object into a DatabaseVideo, which is specialized for handling video analysis in a database context.

        Args:
            file (QuetzalFile): The QuetzalFile object to be converted into a DatabaseVideo.

        Returns:
            DatabaseVideo: A new DatabaseVideo object initialized with the properties from the provided QuetzalFile.
        """
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
        """
        Converts frames from an associated DatabaseVideo to fit the requirements of a QueryVideo. This method is particularly useful when high-resolution frames from a database video need to be processed or analyzed at a different resolution or frame rate.

        Args:
            verbose (bool): If True, additional logs regarding the conversion process are displayed.

        Returns:
            bool: True if the conversion is successful, False otherwise.
        """
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
        Overrides the get_frames method from the Video superclass to include a step that checks for the existence of frames at the desired fps and resolution. If they do not exist, it attempts to generate them by converting frames from the associated DatabaseVideo.

        Args:
            verbose (bool, optional): If True, logs additional information about the frame retrieval or generation process. Defaults to True.

        Returns:
            list: A list of frame file paths after ensuring they are available at the desired fps and resolution.
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
        """
        Creates a QueryVideo instance from an existing QuetzalFile object. This method facilitates the conversion of a generic file object into a QueryVideo, tailored for analyzing query-type videos within the system.

        Args:
            file (QuetzalFile): The QuetzalFile object to be transformed into a QueryVideo.

        Returns:
            QueryVideo: A new QueryVideo object initialized with the attributes from the given QuetzalFile.
        """
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