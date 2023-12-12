import copy
import os
import re
import subprocess
from glob import glob
from os.path import join
from typing import List


import cv2
import gradio as gr
import numpy as np
import utm
from tqdm import tqdm

### Video Frame Extract
def extract_frames(video_path: str, output_dir: str, fps: int=2, max_size = -1, interpolation='bicubic') -> List[str]:
    """
    Extract frames from a video at a specified frame rate and scale them to a maximum size.

    Args:
    video_path (str): Path to the video file.
    output_dir (str): Directory where the extracted frames will be saved.
    fps (int): Frame rate for extraction. Default is 2.
    max_size (int, optional): Maximum size (width or height) for the extracted frames. Default is -1 (no scaling).
    interpolation (str, optional): Interpolation method used for scaling. Default is 'bicubic'. It can be 'lanczos', 'bicubic', or any ffmpeg scaler options.

    Returns:
    list: List of file paths for the extracted frames.
    """

    if not max_size:
        max_size = -1

    # Use the 'scale' filter to set the width or height to 'max_size', while preserving the aspect ratio.
    scale_filter = f"scale='if(gt(iw,ih),{max_size},-1)':'if(gt(iw,ih),-1,{max_size})':flags={interpolation}"

    # Construct the ffmpeg command with both the 'fps' and 'scale' filters.
    command = [
        "ffmpeg", 
        "-i", video_path, 
        "-qscale:v", "2",
        "-vf", f"{scale_filter},fps={fps}", 
        f"{output_dir}/frame%05d.jpg"
    ]
    subprocess.run(command, capture_output=True)

    return glob(join(output_dir, "*.jpg"))

def dms_to_decimal(dms_str: str) -> str:
    """
    Convert a string from DMS (Degrees, Minutes, Seconds) format to decimal format.

    Args:
    dms_str (str): A string representing coordinates in DMS format.

    Returns:
    str: The converted coordinate in decimal format.
    """
    # Extract degrees, minutes, and seconds using regex
    match = re.match(r'(-?\d+) deg (\d+)' + r"' " + r'([\d.]+)"', dms_str)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    
    degrees, minutes, seconds = map(float, match.groups())
    is_negative = degrees < 0 or "S" in dms_str or "W" in dms_str
    decimal_degrees = abs(degrees) + (minutes / 60) + (seconds / 3600)
    result = -decimal_degrees if is_negative else decimal_degrees
    
    return f"{result:3.5f}"


def parse_exiftool_output(output: str) -> List[str]:
    """
    Parse 'TimeStamp', 'Latitude', 'Longitude', and 'Elevation' from exiftool output from Anafi Ai Video footage

    Args:
    output (str): The output string from exiftool.

    Returns:
    list: A list of dictionaries with frame information.
    """
    frames = []
    frame = {}
    for line in output.splitlines():
        if "Sample Time" in line and frame:
            frames.append(frame)
            frame = {}
            time = line.split(":")[1].strip()
            frame['TimeStamp'] = time
        if "GPS Latitude" in line:
            lat_dms = line.split(":")[1].strip()
            frame['Latitude'] = dms_to_decimal(lat_dms)
        if "GPS Longitude" in line:
            lon_dms = line.split(":")[1].strip()
            frame['Longitude'] = dms_to_decimal(lon_dms)
        if "Elevation" in line:
            frame['Elevation'] = line.split(":")[1].strip()
    if frame:
        frames.append(frame)
    return frames

def extract_gps_for_frames(video_path: str, output_directory: str) -> np.ndarray:
    """
    Extract GPS information for frames in a given directory based on exiftool output from a video.

    Args:
    video_path (str): Path to the video file.
    output_directory (str): Directory where the frames are stored.

    Returns:
    numpy.ndarray: An array containing GPS coordinates (UTM Easting and Northing) for each frame.
    """

    # Step 1: Extract exiftool output
    command = ["exiftool", "-ee", video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 1, result.stderr
    gps_frames = parse_exiftool_output(result.stdout)
    img_frames = glob(join(output_directory, "*.jpg"))

    # Step 3: Obtain GPS value (UTM Easting, UTM Northing) for each img_frames
    num_frames = len(img_frames)

    # Align the GPS index to the extracted frame's index
    step = int(len(gps_frames) / len(img_frames) + 0.5)
    gps_frame_idx= list(range(0, len(gps_frames), step))
    diff = num_frames - len(gps_frame_idx)
    while diff > 0:
        gps_frame_idx.append(gps_frame_idx[-1])
        diff = diff - 1
    
    db_gps = np.zeros((num_frames, 2))
    for index, img_frame in enumerate(img_frames):
        latitude = gps_frames[gps_frame_idx[index]].get('Latitude')
        longitude = gps_frames[gps_frame_idx[index]].get('Longitude')
        easting, northing, _zone_number, _zone_letter = utm.from_latlon(float(latitude), float(longitude))
        db_gps[index] = [easting, northing]

    return db_gps

def assert_absolute_path(file_path: str):
    assert os.path.isabs(file_path), f"'{file_path}' is not an absolute path"

if __name__=="__main__":
    from tqdm import tqdm
    max_size = 1024
    database_video = "P0190019.MP4"
    query_video = "P0200020.MP4"

    # Configure folders
    datasets_folder = join(os.curdir, "AnyLoc", "datasets_vg", "datasets")
    dataset_name = "mil19_1fps"
    dataset_folder = join(datasets_folder, dataset_name)

    raw_data_folder = join(datasets_folder, dataset_name, "raw_video")
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(raw_data_folder, exist_ok=True)
    os.makedirs(join(dataset_folder, "images", "test"), exist_ok=True)

    dst_database_folder = join(dataset_folder, "images", "test", "database")
    os.makedirs(dst_database_folder, exist_ok=True)
    dst_queries_folder = join(dataset_folder, "images", "test", "queries")
    os.makedirs(dst_queries_folder, exist_ok=True)

    db_video_path = join(raw_data_folder, database_video)
    query_video_path = join(raw_data_folder, query_video)
    print(db_video_path)
    print(dst_database_folder)

    img_frames = glob(join(dst_database_folder, "*.jpg"))
    print(img_frames)

    for i in tqdm([1]):
        if len(glob(join(dst_database_folder, "*.jpg"))) == 0:
            input_frames = extract_frames(db_video_path, dst_database_folder, max_size = 1024, fps = 6)
            db_gps = extract_gps_for_frames(db_video_path, dst_database_folder)
            np.save(f"{dataset_folder}/db_gps.npy", db_gps)
