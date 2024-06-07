from abc import ABC, abstractmethod
from typing import NewType, TypeAlias
from quetzal.dtos.video import Video
import math
from haversine import haversine, Unit
import re
import subprocess
import utm
from glob import glob
from os.path import join
import numpy as np
from pathlib import Path
import pickle


Latitude: TypeAlias = float
Longitude: TypeAlias = float
Easting: TypeAlias = float
Northing: TypeAlias = float
# ZoneNumber: TypeAlias = int
# ZoneLetter: TypeAlias = str
Quaternion: TypeAlias = tuple[float, float, float, float] # W,X,Y,Z
Elevation: TypeAlias = float # meter
Utm = NewType("Utm", tuple[Easting, Northing])
GpsPoint = NewType("GpsPoint", tuple[Latitude, Longitude])


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
    return f"{result:3.6f}"

def quaternion_to_euler(w, x, y, z):
    """
    Convert a quaternion into euler angles (yaw, pitch, roll) in radians.
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return yaw_z, pitch_y, roll_x  # Return yaw, pitch, roll in radians


def calculate_look_at_gps(latitude, longitude, quaternion, elevation, camera_angle_degrees) -> tuple[Latitude, Longitude]:
        """
        Calculate the GPS position of the point the camera is looking at on the ground.
        Camera angle is measured from the forward direction (0 degrees forward, 90 degrees down)
        """
        EARTH_RADIUS = 6371000  # Earth's radius in meters
        
        # Convert quaternion to euler angles to get yaw
        yaw, _, _ = quaternion_to_euler(*quaternion)
        
        camera_angle_radians = math.radians(camera_angle_degrees)
        pitch = math.pi / 2 - camera_angle_radians  # Adjust pitch for the camera's angle
        
        # Assuming the camera's angle affects the pitch directly for simplicity
        horizontal_distance = elevation / math.tan(pitch)
        m_per_deg_lat = 2 * math.pi * EARTH_RADIUS / 360
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(latitude))
        
        delta_lat = (horizontal_distance * math.cos(yaw)) / m_per_deg_lat
        delta_lon = (horizontal_distance * math.sin(yaw)) / m_per_deg_lon
        
        new_latitude = latitude + delta_lat
        new_longitude = longitude + delta_lon
        
        return new_latitude, new_longitude


def find_frames_within_radius(db_gps, query_point, radius=5):
    """
    Finds frame indices within a specified radius of a query GPS location.

    Args:
        db_gps (numpy.ndarray): Array of shape (num_frames, 2) containing
            latitude/longitude coordinates for each frame.
        query_latitude (float): Latitude of the query location.
        query_longitude (float): Longitude of the query location.
        radius (float, optional): Search radius in meters. Defaults to 5.

    Returns:
        list: List of frame indices within the specified radius.
    """
    frame_indices = []
    closest = None
    closest_idx = -1
    
    for i, (latitude, longitude) in enumerate(db_gps):
        frame_point = (latitude, longitude)
        distance = haversine(query_point, frame_point, unit=Unit.METERS)

        if distance <= radius:
            frame_indices.append(i)
            if not closest or distance < closest:
                closest = distance
                closest_idx = i

    return frame_indices, closest_idx

class AbstractGPS(ABC):
    
    @abstractmethod
    def extract_gps(self) -> tuple[list[GpsPoint], list[Utm], list[Quaternion], list[Elevation]]:
        pass
    
    @abstractmethod
    def get_gps(self) -> list[GpsPoint]:
        pass
    
    @abstractmethod
    def get_look_at_gps(self) -> list[GpsPoint]:
        pass
    
    
    
class AnafiGPS(AbstractGPS):
    
    GPS_DATA_FILE = "anafi_gps_data.pkl"
    
    def __init__(self, video: Video):
        self.video = video
        self.gps = None
        self.utm = None
        self.quaternion = None
        self.elevation = None
        
    def extract_gps(self):
        if self.gps:
            return self.gps, self.utm, self.quaternion, self.elevation
        
        gps_path = Path(self.video.dataset_dir).parent / self.GPS_DATA_FILE
        if gps_path.exists():
            with open(gps_path, 'rb') as f:
                self.gps, self.utm, self.quaternion, self.elevation = pickle.load(f)  
        else:
            self.gps, self.utm, self.quaternion, self.elevation = self._extract_gps()
            with open(gps_path, 'wb') as f:
                pickle.dump((self.gps, self.utm, self.quaternion, self.elevation), f)
        
        return self.gps, self.utm, self.quaternion, self.elevation
    
    def get_gps(self) -> list[GpsPoint]:
        if self.gps is None:
            self.extract_gps()
            
        return self.gps
    
    def get_look_at_gps(self, camera_angle: int=45) -> list[GpsPoint]:
        """
        camera_angle (deg): looking forward = 0 deg, looking down = 90 deg
        """
        if self.gps == None:
            self.extract_gps()
            
        gps_look_at = [calculate_look_at_gps(gps[0], gps[1], quat, elev, camera_angle) for gps, quat, elev in zip(self.gps, self.quaternion, self.elevation)]
        return gps_look_at
        
        
    def _parse_exiftool_output(self, output):
        """
        parse 'TimeStamp', 'Latitude', 'Longitude', and 'Elevation' from exiftool output from Anafi Ai Video footage
        return list of ditionary for each frame
        """

        latitude = []
        longitude = []
        quaternion = []
        elavation = []
        
        for line in output.splitlines():
            if "GPS Latitude" in line:
                lat_dms = line.split(":")[1].strip()
                latitude.append(dms_to_decimal(lat_dms))
            if "GPS Longitude" in line:
                lon_dms = line.split(":")[1].strip()
                longitude.append(dms_to_decimal(lon_dms))
            if "Drone Quaternion" in line:
                quaternion.append(line.split(":")[1].strip().split(" "))
            if "Elevation" in line:
                elavation.append(line.split(":")[1].strip().split(" ")[0])
        
        frames = list()
        for lat, long, quat, el in zip(latitude[:-1], longitude[:-1], quaternion, elavation):
            frames.append({'Latitude':lat, 'Longitude':long, 'Quaternion':quat, 'Elevation':el})
            
        return frames
        
    def _extract_gps(self):            
        """
        Extract corresponding gpu informations for the frames in output_directory
        use after "extract_frames"
        """
        video_path  = self.video._abs_path
        frames_dir = self.video.dataset_dir

        # Step 1: Extract exiftool output
        command = ["exiftool", "-ee", "-GPSLatitude", "-GPSLongitude", "-DroneQuaternion", "-Elevation", video_path]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 1, result.stderr

        gps_frames = self._parse_exiftool_output(result.stdout)
        img_frames = glob(join(frames_dir, "*.jpg"))
        
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
        db_utm = np.zeros((num_frames, 2))
        db_quaternion = np.zeros((num_frames, 4))
        db_elevation = np.zeros(num_frames)
        
        for index, img_frame in enumerate(img_frames):
            frame = gps_frames[gps_frame_idx[index]]
            latitude = frame.get('Latitude')
            longitude = frame.get('Longitude')
            easting, northing, zone_number, zone_letter = utm.from_latlon(float(latitude), float(longitude))
                    
            db_gps[index] = [latitude, longitude]
            db_utm[index] = [easting, northing]
            db_quaternion[index] = [float(val) for val in frame.get('Quaternion')]
            db_elevation[index] =  float(frame.get('Elevation'))

        return db_gps, db_utm, db_quaternion, db_elevation