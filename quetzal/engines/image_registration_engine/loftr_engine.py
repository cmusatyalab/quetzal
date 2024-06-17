import os
from os.path import join
import torch
from kornia.feature import LoFTR
import cv2

from functools import lru_cache
import logging
from quetzal.dtos.video import Video
from typing import Literal, List, NewType
from quetzal.engines.engine import AbstractEngine


logging.basicConfig()
logger = logging.getLogger("LoFTR_Engine")
logger.setLevel(logging.DEBUG)

FilePath = NewType("FilePath", str)


class LoFTREngine(AbstractEngine):
    def __init__(
        self,
        query_video: Video,
        device: torch.device=torch.device("cuda:0"),
        db_name: str="",
        pretrained: Literal["outdoor", "indoor"] = "outdoor"
    ):
        """
        Assumes Using GPU (cuda)
        """

        self.name = "Image Alignment - LoFTR"
        self.save_dir = join(
            query_video.dataset_dir,
            db_name + "_LoFTR",
        )
        os.makedirs(self.save_dir, exist_ok=True)

        ## Loading model
        self.device = device
        self.model = LoFTR(pretrained=pretrained)
        self.model = self.model.eval().to(device)
        
    def analyze_video(self, video: Video):
        """Return True if no further real-time analysis required"""
        
        pass
    
    def _get_transformation_matrix(self, query_image, databse_image):
        img0_raw = cv2.imread(query_image)
        img1_raw = cv2.imread(databse_image)
        orig_w, orig_h = img0_raw.shape[1], img0_raw.shape[0]


        img0 = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2GRAY)
        img0 = cv2.resize(img0, (img0.shape[1]//8*8, img0.shape[0]//8*8))  # input size shuold be divisible by 8
        img0 = torch.from_numpy(img0)[None][None].to(self.device) / 255.

        img1 = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (img1.shape[1]//8*8, img1.shape[0]//8*8))
        img1 = torch.from_numpy(img1)[None][None].to(self.device) / 255.

        resized = False
        if (img0.shape[1] != orig_w or img0.shape[0] != orig_h):
            resized = True
            img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8

        batch = {'image0': img0, 'image1': img1}
        # Inference with LoFTR and get prediction
        
        with torch.no_grad():
            data = self.model(batch)
            mkpts0 = data["keypoints0"].cpu().numpy()
            mkpts1 = data["keypoints1"].cpu().numpy()
                    
        H, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC)
        
        return H
    
    def _generate_aligned_from_H(self, query_image, H, save_file_path):
        
        img0_raw = cv2.imread(query_image)
        orig_w, orig_h = img0_raw.shape[1], img0_raw.shape[0]
        
        aligned_img = cv2.warpPerspective(img0_raw, H, (img0_raw.shape[1], img0_raw.shape[0]))
        
        # Save the image
        cv2.imwrite(save_file_path, aligned_img)


    def _generate_aligned_images(
        self, query_image: FilePath, database_image: FilePath, save_file_path: FilePath
    ):
        """
        Aligns a query image with a corresponding image from the database using the LoFTR model and saves the aligned image.

        The method reads both query and database images, converts them to grayscale, and resizes them to be divisible by 8,
        as required by the LoFTR model. It then performs feature matching and geometric transformation to align the query image
        with the database image. The aligned image is resized back to the original dimensions of the query image if necessary
        and saved to the specified path.

        Args:
            query_image (FilePath): The file path of the query image to be aligned.
            database_image (FilePath): The file path of the corresponding database image.
            save_file_path (FilePath): The file path where the aligned query image should be saved.
        """
        img0_raw = cv2.imread(query_image)
        img1_raw = cv2.imread(database_image)
        orig_w, orig_h = img0_raw.shape[1], img0_raw.shape[0]


        img0 = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2GRAY)
        img0 = cv2.resize(img0, (img0.shape[1]//8*8, img0.shape[0]//8*8))  # input size shuold be divisible by 8
        img0 = torch.from_numpy(img0)[None][None].to(self.device) / 255.

        img1 = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (img1.shape[1]//8*8, img1.shape[0]//8*8))
        img1 = torch.from_numpy(img1)[None][None].to(self.device) / 255.

        resized = False
        if (img0.shape[1] != orig_w or img0.shape[0] != orig_h):
            resized = True
            img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8

        batch = {'image0': img0, 'image1': img1}
        # Inference with LoFTR and get prediction
        with torch.no_grad():
            data = self.model(batch)
            mkpts0 = data["keypoints0"].cpu().numpy()
            mkpts1 = data["keypoints1"].cpu().numpy()
                    
        H, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC)
        aligned_img = cv2.warpPerspective(img0_raw, H, (img0_raw.shape[1], img0_raw.shape[0]))

        if resized:
            aligned_img = cv2.resize(aligned_img, (orig_w, orig_h))
        
        # Save the image
        cv2.imwrite(save_file_path, aligned_img)
            
    @lru_cache(maxsize=None)
    def process(self, file_path: tuple[FilePath, FilePath]):
        """Process list of files in file_path

        Return an resulting file."""
        if not isinstance(file_path, tuple):
            logger.error(
                "Invalid input. Input should be tuple. Received: " + str(file_path)
            )
            return None

        if len(file_path) != 2:
            logger.error(
                "Invalid input. Input should have len = 2. Received: " + str(file_path)
            )
            return None

        # Extract the filename from the query path
        query_filename = os.path.basename(file_path[0])

        # Create the complete save path
        save_file_path = os.path.join(self.save_dir, query_filename)

        if not os.path.exists(save_file_path):
            # if not self.cached:
            self._generate_aligned_images(file_path[0], file_path[1], save_file_path)

        return (save_file_path, file_path[1])

    def end(self):
        """Save state in save_path."""
        return None

    def save_state(self, save_path: FilePath):
        return None


if __name__ == "__main__":
    engine = LoFTREngine()


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