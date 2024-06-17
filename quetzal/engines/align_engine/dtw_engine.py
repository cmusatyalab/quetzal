import os
from os.path import join
import torch
import subprocess
from groundingdino.util.inference import Model
import supervision as sv

import cv2

import logging
from quetzal.engines.engine import AlignmentEngine, FrameMatch, WarpedFrame
from quetzal.align_frames import align_frame_pairs, align_video_frames
from quetzal.dtos.video import Video

logging.basicConfig()
logger = logging.getLogger("DTW Engine")
logger.setLevel(logging.DEBUG)

class DTWEngine(AlignmentEngine):
    name = "grounding_sam"
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
    ):
        """
        Assumes Using GPU (cuda)
        """

        ## Loading model
        self.device = device
    
    def align_frame_list(
        self, database_video: Video, query_video: Video, overlay: bool
    ) -> tuple[FrameMatch, WarpedFrame]:
        ## Load DTW and VLAD Features ##
        
        db_frame_list = database_video.get_frames()
        query_frame_list = query_video.get_frames()
        warp_query_frame_list = query_frame_list

        if not overlay:
            matches = align_video_frames(
                database_video=database_video,
                query_video=query_video,
                torch_device=self.device,
            )
        else:
            matches, warp_query_frame_list = align_frame_pairs(
                database_video=database_video,
                query_video=query_video,
                torch_device=self.device,
            )
        
        return matches, warp_query_frame_list
        
if __name__ == "__main__":
    engine = DTWEngine()
