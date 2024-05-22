import logging
import os
import re
import threading
from glob import glob
from os.path import join
from typing import Literal, List
from typing import Tuple, TypeAlias, NewType

import cv2
import numpy as np
import natsort
from PIL import Image
import shutil

# from quetzal.dtos import QuetzalFile
from quetzal.utils.video_tools import extract_frames
import time
logging.basicConfig()


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


# class QuetzalVideo(QuetzalFile):
#     def __init__(
#         self,
#         path: str,
#         root_dir: str,
#         metadata_dir: str,
#         video_type: Literal["database", "query"],
#         fps: int = 2,
#         resolution: int = 1024,
#         user: str = None,
#         mode: Literal["user", "shared"] = "user",
#     ):
#         super().__init__(
#             path=path,
#             root_dir=root_dir,
#             metadata_dir=metadata_dir,
#             user=user,
#         )
        


class Video:
    """
    A class to manage video processing for drone footage.

    Attributes:
        root_datasets_dir (str): The root directory for datasets.
        project_name (str): The name of the specific project the drone video was taken.
        video_name (str): The name of the video file.
        video_type (Literal["database", "query"]): The type of the video, either 'database' or 'query'.
        fps (int): The frames per second at which to process the video.
        resolution (int): The resolution for the processed frames.
        ...
    """

    def __init__(
        self,
        datasets_dir: str,
        project_name: str,
        video_name: str,
        video_type: Literal["database", "query"],
        fps: int = 2,
        resolution: int = 1024,
        metadata_dir: str = None,
    ):
        """
        Initialize the Video object with dataset directory, project, video name, type, fps, and resolution.

        Args:
            datasets_dir (str): The root directory for datasets.
            project_name (str): The name of the specific project.
            video_name (str): The name of the video file.
            video_type (Literal["database", "query"]): The type of the video.
            fps (int): Frames per second for video processing.
            resolution (int): The resolution for the processed frames.

            Dataset structure
            root_datsets_dir/
            |
            ├── project_name/
            |   ├── video_name.mp4
            |   ├── ...
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

        assert video_type in [
            "database",
            "query",
        ], "video_type must be 'database' or 'query'"

        self.root_datasets_dir = os.path.abspath(datasets_dir)
        self.project_name = project_name
        self.video_name = video_name
        self.video_type: Literal["database", "query"] = video_type
        self.metadata_dir = metadata_dir
        
        if self.metadata_dir:
            self.metadata_dir = os.path.abspath(self.metadata_dir)
            self.dataset_dir = os.path.abspath(join(
                self.metadata_dir,
                self.project_name,
                self.video_type,
                os.path.splitext(self.video_name)[0],
            ))
        else:
            self.dataset_dir = os.path.abspath(join(
                self.root_datasets_dir,
                self.project_name,
                self.video_type,
                os.path.splitext(self.video_name)[0],
            ))
            
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.fps = fps
        self.resolution = resolution

        self.gps = None
        self.logger = logging.getLogger("Video_" + video_name)
        self.logger.setLevel(logging.DEBUG)

        self._lock = threading.Lock()

        # Check if the video file exists
        raw_video_file = self.get_raw_video_file()
        if not os.path.exists(raw_video_file):
            raise FileNotFoundError(f"Video file not found: {raw_video_file}")

        # Initialize the camera and check if it's a valid video file
        self.cam = cv2.VideoCapture(raw_video_file)
        if not self.cam.isOpened():
            raise ValueError(f"Failed to open video file: {raw_video_file}")

        self.native_fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.orig_width, self.orig_height = int(self.cam.get(3)), int(self.cam.get(4))

        self.video_start = 0
        self.video_end = None

    def get_video_name(self):
        return os.path.splitext(self.video_name)[0]

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
        return int(frame_time * self.fps + 0.5)

    def get_frame_time(self, frame_index: int) -> float:
        """
        Converts a frame index to a time point in the video.

        Args:
            frame_index (int): The index of the frame.

        Returns:
            float: Time in seconds where the frame is located in the video.
        """
        return frame_index / self.fps

    def get_raw_frame_at(self, idx: int) -> np.ndarray:
        """
        Retrieves the raw frame at a given index from the video.

        Args:
            idx (int): The frame index.

        Returns:
            ndarray: The raw frame as a numpy array in RGB format, or None if retrieval fails.
        """

        orig_frame_idx = int(idx / self.fps * self.native_fps + 0.5)
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
        rv, frame = self.cam.read()

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
        index = sec * self.fps
        self.video_start = index

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
        index = sec * self.fps
        if index > self.video_start:
            self.video_end = index
        else:
            minute = int(self.video_start / self.fps / 60)
            sec = int(self.video_start / self.fps) % 60
            logging.error(
                f"Video-end Should be greator than video-start = {minute}:{sec}"
            )
            self.video_end = None

    def get_video_start_idx(self) -> int:
        """
        Gets the starting frame index of the video for processing.

        Returns:
            int: The starting frame index.
        """
        return self.video_start

    def get_video_end_idx(self) -> int:
        """
        Gets the ending frame index of the video for processing.

        Returns:
            int: The ending frame index, or None if not set.
        """
        return self.video_end

    def get_fps(self) -> int:
        """
        Retrieves the frames per second setting of the video.

        Returns:
            int: The frames per second.
        """
        return self.fps

    def get_resolution(self) -> int:
        """
        Retrieves the resolution setting of the video.

        Returns:
            int: The resolution of the video.
        """
        return self.resolution

    def set_fps(self, fps: int):
        """
        Sets the frames per second for video processing.

        Args:
            fps (int): The frames per second value to set.
        """
        self.fps = fps

    def set_resolution(self, res: int):
        """
        Sets the resolution for video processing.

        Args:
            res (int): The resolution value to set.
        """
        self.resolution = res

    def get_dataset_dir(self) -> str:
        """
        Retrieves the directory where the dataset frames are stored.

        Returns:
            str: The path to the dataset directory.
        """
        return join(self.dataset_dir, f"frames_{self.fps}_{self.resolution}")

    def get_raw_video_file(self) -> str:
        """
        Retrieves the file path of the raw video file.

        Returns:
            str: The file path of the raw video.
        """
        return join(
            self.root_datasets_dir, self.project_name, self.video_name
        )

    def get_avaliablity(self) -> List[tuple]:
        """
        Checks the availability of different combinations of fps and resolution for which the frames are extracted.

        Returns:
            list[tuple]: A list of tuples (fps, resolution) available in the dataset.
        """
        dir_list = [
            d
            for d in os.listdir(self.dataset_dir)
            if os.path.isdir(join(self.dataset_dir, d))
        ]

        if len(dir_list) == 0:
            return []

        return [
            extract_fps_res(dir)
            for dir in dir_list
            if self.is_dir_valid(join(self.dataset_dir, dir))
        ]

    def is_dir_valid(self, dir: str) -> bool:
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

        fps = self.fps
        resolution = self.resolution

        target_dir = self.get_dataset_dir()
        frame_count_file = join(target_dir, "frame_counts.txt")

        loaded = False
        with self._lock:
            if not self.is_dir_valid(target_dir):
                if verbose:
                    self.logger.info(
                        f"Frames in fps={fps}, res={resolution} do not exist. Generating..."
                    )
                os.makedirs(target_dir, exist_ok=True)
                input_frames = extract_frames(
                    self.get_raw_video_file(), target_dir, max_size=resolution, fps=fps
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

        self.frame_len = len(input_frames)
        idx_start = self.video_start
        if self.video_end == None:
            idx_end = len(input_frames)
        else:
            idx_end = self.video_end
        return natsort.natsorted(input_frames)[
            idx_start : min(idx_end, len(input_frames))
        ]

    def get_frame_len(self) -> int:
        """
        Retrieves the total number of frames in the dataset directory.

        Returns:
            int: The number of frames, or -1 if the frame count is not available.
        """

        target_dir = self.get_dataset_dir()
        frame_count_file = join(target_dir, "frame_counts.txt")

        with self._lock:
            if os.path.exists(frame_count_file):
                return frame_count(frame_count_file)
            else:
                return -1


class DatabaseVideo(Video):
    """
    A subclass of Video for processing database-type videos.

    Inherits all attributes and methods from Video.
    """

    def __init__(
        self, 
        datasets_dir: str, 
        project_name: str, 
        video_name: str,
        metadata_dir: str = None,
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
            datasets_dir=datasets_dir,
            project_name=project_name,
            video_name=video_name,
            video_type="database",
            fps=6,
            resolution=1024,
            metadata_dir=metadata_dir
        )

    # def get_frames(self, fps = 6, resolution = 1024):
    #     return super().get_frames(fps, resolution)


class QueryVideo(Video):
    """
    A subclass of Video for processing query-type videos.

    Inherits all attributes and methods from Video.
    """

    def __init__(
        self, 
        datasets_dir: str, 
        project_name: str, 
        video_name: str,
        metadata_dir: str = None,
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
            datasets_dir=datasets_dir,
            project_name=project_name,
            video_name=video_name,
            video_type="query",
            fps=2,
            resolution=1024,
            metadata_dir=metadata_dir
        )
        
    def _convert_from_database(self, verbose):
        database_video = DatabaseVideo(
            self.root_datasets_dir, 
            self.project_name,
            self.video_name,
            self.metadata_dir     
        )
        database_target = database_video.get_dataset_dir()
        if not database_video.is_dir_valid(database_target):
            return False
        
        if verbose:
            self.logger.info(
                f"Extracing Frames in fps={self.fps}, res={self.resolution} from Database..."
            )
            
        target_dir = self.get_dataset_dir()
        os.makedirs(target_dir, exist_ok=True)
        
        database_frames = database_video.get_frames()
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

        fps = self.fps
        resolution = self.resolution

        target_dir = self.get_dataset_dir()
        frame_count_file = join(target_dir, "frame_counts.txt")

        loaded = False
        with self._lock:
            if not self.is_dir_valid(target_dir):
                if verbose:
                    self.logger.info(
                        f"Frames in fps={fps}, res={resolution} do not exist. Generating..."
                    )
                os.makedirs(target_dir, exist_ok=True)
                
                if self._convert_from_database(verbose):                    
                    input_frames = Video.get_frame_list(target_dir)
                    loaded = True
                
                else:
                    input_frames = extract_frames(
                        self.get_raw_video_file(), target_dir, max_size=resolution, fps=fps
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

        self.frame_len = len(input_frames)
        idx_start = self.video_start
        if self.video_end == None:
            idx_end = len(input_frames)
        else:
            idx_end = self.video_end
        return natsort.natsorted(input_frames)[
            idx_start : min(idx_end, len(input_frames))
        ]


# TODO
class LiveQueryVideo(Video):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, video_type="query", **kwargs)

    def get_frames(self, fps=2, resolution=1024):
        super().get_frames(fps, resolution)
