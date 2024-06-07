import os
from os.path import join
import torch
import subprocess
from groundingdino.util.inference import Model
import supervision as sv

import cv2

import logging
from quetzal.engines.engine import AlignmentEngine, FrameMatch, WarpedFrame, AbstractEngine
from quetzal.align_frames import align_frame_pairs, align_video_frames
from quetzal.dtos.video import Video
from quetzal.dtos.gps import AnafiGPS, find_frames_within_radius, AbstractGPS, GpsPoint
import faiss
import torch
import torch.nn.functional as F
from tqdm import tqdm
from stqdm import stqdm
from quetzal.engines.vpr_engine.anyloc_engine import AnyLocEngine
import numpy as np
import pickle
from pathlib import Path
from typing import TypeAlias
import time

logging.basicConfig()
logger = logging.getLogger("RealtimeAlignment Engine")
logger.setLevel(logging.DEBUG)

def is_transformation_acceptable(H, max_translation=200, min_scale=0.2, max_scale=3.0):
    """
    Check if the homography matrix H suggests an acceptable transformation.
    This simplified check focuses on translation magnitude and scale factor.
    """
    # Decompose H to get scale factor (approximated by determinant) and translation
    scale_factor = np.linalg.det(H[:2, :2]) ** 0.5
    translation_magnitude = np.linalg.norm(H[:2, 2])
    
    if (translation_magnitude > max_translation or
        scale_factor < min_scale or scale_factor > max_scale or
        np.isnan(scale_factor)
        ):

        return False
    
    return True

def calculate_blackout_area(image):
    # Assuming a black pixel has [0, 0, 0] in all channels
    black_pixels = np.all(image == 0, axis=-1)
    return np.sum(black_pixels)

FilePath256: TypeAlias = str
FilePath512: TypeAlias = str

class RealtimeAlignmentEngine(AlignmentEngine, AbstractEngine):
    name = "grounding_sam"
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
        database_video: Video = None,
        database_gps: AbstractGPS = None,
        camera_angle: int = 45,
        gps_look_at: bool = True,
        query_num: int = 10,
    ):
        """
        Assumes Using GPU (cuda)
        """

        ## Loading model
        self.device = device
        self.database_video = database_video
        self.database_gps = database_gps
        self.camera_angle = camera_angle
        self.query_num = query_num
        
        
        
        if database_video and database_gps:
            
            logger.debug("Loading AnylocEngine")
            
            database_video.resolution = 512
            self.db_frames_512 = database_video.get_frames()
            database_video.resolution = 256
            
            if gps_look_at:
                self.db_gps_look_at: list[GpsPoint] = database_gps.get_look_at_gps(camera_angle=camera_angle)
            else:
                self.db_gps_look_at: list[GpsPoint] = database_gps.get_gps()

            self.anyloc_256 = AnyLocEngine(
                database_video=database_video,
                query_video=None,
                max_img_size=256,
            )
            self.anyloc_256.load_models()
            self.db_256 = self.anyloc_256.get_database_vlad()
            
            self.res = faiss.StandardGpuResources()
            self.orb = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.db_vlad = F.normalize(self.db_256)
            
            ex_image = self.db_frames_512[0]
            ex_image = cv2.imread(ex_image)
            img_shape = ex_image.shape
            self.img_pixels = img_shape[0] * img_shape[1]
            
        self.frame_match = list()
        
        logger.debug("Done Loading")
        
        
    def is_video_analyzed(video) -> bool:
        """Return True if no further real-time analysis required"""

        return False
    
    
    # def process(self, file_path: list[(FilePath256, FilePath512), AbstractGPS]):
    #     """take (file_path, AbstractGps)"""
        
    #     (query_frame, query_frame_512), query_gps_look_at = file_path
        
    #     query = self.anyloc_256._get_vlad(query_frame)
            
    #     db_idx_selected, _= find_frames_within_radius(db_gps=self.db_gps_look_at, query_point=query_gps_look_at, radius=20)

    #     db_256_gps = np.array([self.db_vlad[idx] for idx in db_idx_selected])
    #     db_index = faiss.IndexFlatIP(db_256_gps.shape[1])
    #     db_index: faiss.IndexFlatIP = faiss.index_cpu_to_gpu(self.res, 0, db_index)
    #     db_index.add(db_256_gps)
        
    #     _distance, db_idx_from_vlad = db_index.search(query, self.query_num)
    #     db_idx_from_vlad = [db_idx_selected[idx] for idx in db_idx_from_vlad[0]]
        
    #     db_frames_input = [self.db_frames_512[idx] for idx in db_idx_from_vlad]
        
    #     blackout_area_list = []
    #     magnitudes = []
    #     img_query = cv2.imread(query_frame_512)
    #     kp_query, des_query = self.orb.detectAndCompute(img_query, None)
    #     for i, db in enumerate(db_frames_input):
    #     # Load images
    #         img_db = cv2.imread(db)

    #         # Detect keypoints and descriptors with ORB
    #         kp_db, des_db = self.orb.detectAndCompute(img_db, None)

    #         # Match descriptors
    #         matches = self.bf.match(des_db, des_query)
    #         matches = sorted(matches, key = lambda x:x.distance)
    #         points_db = np.float32([kp_db[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #         points_query = np.float32([kp_query[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
    #         # Find homography
    #         H, _ = cv2.findHomography(points_db, points_query, cv2.RANSAC)
    #         if H is not None and is_transformation_acceptable(H):
    #             aligned_img = cv2.warpPerspective(img_db, H, (img_query.shape[1], img_query.shape[0]))
    #             blackout_area = calculate_blackout_area(aligned_img)
    #             blackout_area_list.append(blackout_area)
    #             magnitudes.append((i, blackout_area))
    #         else:
    #             blackout_area = self.img_pixels
    #             blackout_area_list.append(blackout_area)
    #             magnitudes.append((i, blackout_area))

    #     # Sort by magnitudes
    #     magnitudes.sort(key=lambda x: x[1])  
    #     if magnitudes[0][1] == self.img_pixels:
    #         aligned_db_idx = db_idx_from_vlad[0]
    #     else:
    #         aligned_db_idx = db_idx_from_vlad[magnitudes[0][0]]
        
    #     ## NEED CHANGE
    #     query_idx = int(query_frame[-9:-4]) - 1
    #     self.frame_match.append((query_idx, aligned_db_idx))
        
    #     # print("Matched", (self.query_idx, aligned_db_idx))
    #     return (query_idx, aligned_db_idx)
    
    def resize_image_keep_aspect_ratio(self, image_path, max_img_size=512):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        if max((h, w)) >= max_img_size:
            if h == max((h, w)):
                w = int(w * max_img_size / h)
                h = max_img_size
            else:
                h = int(h * max_img_size / w)
                w = max_img_size

            resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            return resized_img
        else:
            return img
    
    def process(self, file_path: list[str, AbstractGPS]):
        """take (file_path, AbstractGps)"""
        # start_0 = time.time()
        # print(file_path)
        query_frame, query_gps_look_at = file_path
        query = self.anyloc_256._get_vlad(query_frame)
        
        # print("Anyloc: ", time.time() - start_0)
        
        # print(self.db_gps_look_at)
        
        # start = time.time()
        
        db_idx_selected, _= find_frames_within_radius(db_gps=self.db_gps_look_at, query_point=query_gps_look_at, radius=20)
        radius = 20
        while not db_idx_selected:
            radius += 60
            db_idx_selected, _= find_frames_within_radius(db_gps=self.db_gps_look_at, query_point=query_gps_look_at, radius=radius)


        db_256_gps = np.array([self.db_vlad[idx] for idx in db_idx_selected])
        db_index = faiss.IndexFlatIP(db_256_gps.shape[1])
        db_index: faiss.IndexFlatIP = faiss.index_cpu_to_gpu(self.res, 0, db_index)
        db_index.add(db_256_gps)
        
        _distance, db_idx_from_vlad = db_index.search(query, self.query_num)
        db_idx_from_vlad = [db_idx_selected[idx] for idx in db_idx_from_vlad[0]]
        
        db_frames_input = [self.db_frames_512[idx] for idx in db_idx_from_vlad]
        
        blackout_area_list = []
        magnitudes = []
        
        # print("GPS Search: ", time.time() - start)

        # start = time.time()
        img_query = self.resize_image_keep_aspect_ratio(query_frame, 512)
        # print("Resize: ", time.time() - start)
        
        # start = time.time()
        kp_query, des_query = self.orb.detectAndCompute(img_query, None)
        for i, db in enumerate(db_frames_input):
        # Load images
            img_db = cv2.imread(db)

            # Detect keypoints and descriptors with ORB
            kp_db, des_db = self.orb.detectAndCompute(img_db, None)

            # Match descriptors
            matches = self.bf.match(des_db, des_query)
            matches = sorted(matches, key = lambda x:x.distance)
            points_db = np.float32([kp_db[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points_query = np.float32([kp_query[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, _ = cv2.findHomography(points_db, points_query, cv2.RANSAC)
            if H is not None and is_transformation_acceptable(H):
                aligned_img = cv2.warpPerspective(img_db, H, (img_query.shape[1], img_query.shape[0]))
                blackout_area = calculate_blackout_area(aligned_img)
                blackout_area_list.append(blackout_area)
                magnitudes.append((i, blackout_area))
            else:
                blackout_area = self.img_pixels
                blackout_area_list.append(blackout_area)
                magnitudes.append((i, blackout_area))

        
        # Sort by magnitudes
        magnitudes.sort(key=lambda x: x[1])  
        if magnitudes[0][1] == self.img_pixels:
            aligned_db_idx = db_idx_from_vlad[0]
        else:
            aligned_db_idx = db_idx_from_vlad[magnitudes[0][0]]
        
        # print("ORB: ", time.time() - start)
        ## NEED CHANGE
        query_idx = int(query_frame[-9:-4]) - 1
        self.frame_match.append((query_idx, aligned_db_idx))
        
        # print("Matched", (query_idx, aligned_db_idx, query_frame))
        # print("Proccess Time: ", time.time() - start_0)
        return (query_idx, aligned_db_idx, query_frame)


    def end(self):
        """Indicate no more input will be processed"""
        pass


    def save_state(self, save_path):
        """Save state in save_path. return None or final result"""
        pass
    
    
    def align_frame_list(
        self, database_video: Video, query_video: Video, overlay: bool, query_num: int = 10
    ) -> tuple[FrameMatch, WarpedFrame]:
        #Load Neseccary Frames  
        
        match_file: Path = query_video.dataset_dir.parent / ("match_with_" + database_video.full_path.stem)
        if match_file.exists():
            with open(match_file, 'rb') as f:
                frame_match, warp_query_frame_list = pickle.load(f)  
            
            return frame_match, warp_query_frame_list
        
        
        database_video.resolution = 512
        db_frames_512 = database_video.get_frames()
        database_video.resolution = 256
        # db_frames_256 = database_video.get_frames()
        
        query_video.resolution = 512
        query_frames_512 = query_video.get_frames()
        query_video.resolution = 256
        query_frames_256 = query_video.get_frames()
        
        db_gps_anafi = AnafiGPS(database_video)
        query_gps_anafi = AnafiGPS(query_video)
        
        db_gps_look_at = db_gps_anafi.get_look_at_gps(camera_angle=45)
        query_gps_look_at = query_gps_anafi.get_look_at_gps(camera_angle=45)
        

        anyloc_256 = AnyLocEngine(
            database_video=database_video,
            query_video=query_video,
            max_img_size=256,
        )
        anyloc_256.load_models()
        db_256 = anyloc_256.get_database_vlad()
                
        res = faiss.StandardGpuResources()
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        db_vlad = F.normalize(db_256)
        
        ex_image = query_frames_512[0]
        ex_image = cv2.imread(ex_image)
        img_shape = ex_image.shape
        img_pixels = img_shape[0] * img_shape[1]
        
        frame_match = list()
        query_idx = 0
        for query_frame in stqdm(query_frames_256, desc="Matching Frames Real-time", backend=True):
            query = anyloc_256._get_vlad(query_frame)
            
            db_idx_selected, _= find_frames_within_radius(db_gps=db_gps_look_at, query_point=query_gps_look_at[query_idx], radius=20)

            db_256_gps = np.array([db_vlad[idx] for idx in db_idx_selected])
            db_index = faiss.IndexFlatIP(db_256_gps.shape[1])
            db_index: faiss.IndexFlatIP = faiss.index_cpu_to_gpu(res, 0, db_index)
            db_index.add(db_256_gps)
            
            _distance, db_idx_from_vlad = db_index.search(query, query_num)
            db_idx_from_vlad = [db_idx_selected[idx] for idx in db_idx_from_vlad[0]]
            
            db_frames_input = [db_frames_512[idx] for idx in db_idx_from_vlad]
            
            blackout_area_list = []
            magnitudes = []
            img_query = cv2.imread(query_frames_512[query_idx])
            kp_query, des_query = orb.detectAndCompute(img_query, None)
            for i, db in enumerate(db_frames_input):
            # Load images
                img_db = cv2.imread(db)

                # Detect keypoints and descriptors with ORB
                kp_db, des_db = orb.detectAndCompute(img_db, None)

                # Match descriptors
                matches = bf.match(des_db, des_query)
                matches = sorted(matches, key = lambda x:x.distance)
                points_db = np.float32([kp_db[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                points_query = np.float32([kp_query[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, _ = cv2.findHomography(points_db, points_query, cv2.RANSAC)
                if H is not None and is_transformation_acceptable(H):
                    aligned_img = cv2.warpPerspective(img_db, H, (img_query.shape[1], img_query.shape[0]))
                    blackout_area = calculate_blackout_area(aligned_img)
                    blackout_area_list.append(blackout_area)
                    magnitudes.append((i, blackout_area))
                else:
                    blackout_area = img_pixels
                    blackout_area_list.append(blackout_area)
                    magnitudes.append((i, blackout_area))

            # Sort by magnitudes
            magnitudes.sort(key=lambda x: x[1])  
            if magnitudes[0][1] == img_pixels:
                aligned_db_idx = db_idx_from_vlad[0]
            else:
                aligned_db_idx = db_idx_from_vlad[magnitudes[0][0]]
            
            frame_match.append((query_idx, aligned_db_idx))
            query_idx += 1
            
        database_video.resolution = 1024
        query_video.resolution = 1024

        warp_query_frame_list = query_video.get_frames()
        
        with open(match_file, 'wb') as f:
            pickle.dump((frame_match, warp_query_frame_list), f)
        
        return frame_match, warp_query_frame_list
        
if __name__ == "__main__":
    engine = RealtimeAlignmentEngine()
