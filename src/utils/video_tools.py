import re
import subprocess
import utm
from glob import glob
from os.path import join
import numpy as np
import cv2
import os
import copy

def extract_frames(video_path, output_dir, fps, max_size=-1):
    """
    Extract image frames from a video using OpenCV (cv2).

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where extracted frames will be saved.
        fps (int): Number of frames per second to extract.
        max_size (int): Maximum size of the longest dimension of the frame. -1 for original size.

    Returns:
        list: A list of file paths to the extracted frames.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Get original video FPS and total frame count
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame interval to match the target FPS
    frame_interval = int(round(original_fps / fps))

    # Read and save frames
    saved_frames = []
    frame_index = 0
    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # Resize frame if max_size is specified
            if max_size > 0:
                h, w = frame.shape[:2]
                scaling_factor = max_size / max(h, w)
                if scaling_factor < 1:
                    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

            # Save frame
            frame_file = join(output_dir, f"frame{save_idx:05d}.jpg")
            cv2.imwrite(frame_file, frame)
            saved_frames.append(frame_file)
            save_idx += 1

        frame_index += 1

    cap.release()
    return saved_frames

### Video Frame Extract
# def extract_frames(video_path, output_dir, fps, max_size = -1, interpolation='bicubic'):
#     """
#     Extract image frames from given video in 'video_path' into 'output_dir'. Calls ffmpeg as subprocess.

#     Default fps is 2. 
#     full-size image will be extracted unless the max_size is specificed. default is None
#     interpolation can be 'lanczos', 'bicubic', or any ffmpeg scaler options. Default is bicubic
#     """

#     if not max_size:
#         max_size = -1

#     # Use the 'scale' filter to set the width or height to 'max_size', while preserving the aspect ratio.
#     scale_filter = f"scale='if(gt(iw,ih),{max_size},-1)':'if(gt(iw,ih),-1,{max_size})':flags={interpolation}"

#     # Construct the ffmpeg command with both the 'fps' and 'scale' filters.
#     command = [
#         "ffmpeg", 
#         "-i", video_path, 
#         "-qscale:v", "2",
#         "-vf", f"{scale_filter},fps={fps}", 
#         f"{output_dir}/frame%05d.jpg"
#     ]
    
#     subprocess.run(command, capture_output=True)

#     return glob(join(output_dir, "*.jpg"))

def dms_to_decimal(dms_str):
    """
    Convert a string in DMS format (DDD° MM' SS.S") to Decimal Degrees.
    """
    # Extract degrees, minutes, and seconds using regex
    match = re.match(r'(-?\d+) deg (\d+)' + r"' " + r'([\d.]+)"', dms_str)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    
    degrees, minutes, seconds = map(float, match.groups())
    
    # Check if degrees is negative to handle W/S coordinates
    is_negative = degrees < 0 or "S" in dms_str or "W" in dms_str
    
    # Convert to decimal format
    decimal_degrees = abs(degrees) + (minutes / 60) + (seconds / 3600)
    
    # Return negative value if it's a W/S coordinate
    result = -decimal_degrees if is_negative else decimal_degrees
    return f"{result:3.5f}"


def parse_exiftool_output(output):
    """
    parse 'TimeStamp', 'Latitude', 'Longitude', and 'Elevation' from exiftool output from Anafi Ai Video footage
    return list of ditionary for each frame
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

def extract_gps_for_frames(video_path, output_directory):
    """
    Extract corresponding gpu informations for the frames in output_directory
    use after "extract_frames"
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

def assert_absolute_path(file_path):
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
