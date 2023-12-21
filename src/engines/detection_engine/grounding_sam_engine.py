import os
from os.path import join
import torch
import subprocess
from groundingdino.util.inference import Model
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import supervision as sv

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from segment_anything import SamPredictor

import cv2

from functools import lru_cache
import logging
from src.video import Video
from typing import Literal, List
from src.engines.engine import AbstractEngine

logging.basicConfig()
logger = logging.getLogger("GroundingSAM Engine")
logger.setLevel(logging.DEBUG)

_ex = lambda x: os.path.realpath(os.path.expanduser(x))
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
grounding_path = join(current_file_dir, "../../external/GroundingDINO")
weight_dir = join(current_file_dir, "../../../weights")

GROUNDING_DINO_CHECKPOINT_PATH: str = _ex(join(weight_dir, "groundingdino_swint_ogc.pth"))
GROUNDING_DINO_CONFIG_PATH: str = _ex(join(grounding_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py"))

BOX_TRESHOLD = 0.25 #0.3
TEXT_TRESHOLD = 0.25

SAM_CHECKPOINT_PATH  = os.path.join(weight_dir, "sam_vit_h_4b8939.pth")
SAM_ENCODER_VERSION = "vit_h"

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
GROUNDING_DINO_CHECKPOINT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"


class GoundingSAMEngine(AbstractEngine):
    def __init__(
        self,
        device: torch.device=torch.device("cuda:0"),
    ):
        """
        Assumes Using GPU (cuda)
        """

        self.name = "Object Detection - GroundingSAM"
        self.save_dir = join(current_file_dir, "../../../tmp")
        os.makedirs(self.save_dir, exist_ok=True)

        if not os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH): 
            logger.info("Downloading weight checkpoint for GroundingDINO")
            self._download_weight(GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CHECKPOINT_URL)

        if not os.path.isfile(SAM_CHECKPOINT_PATH): 
            logger.info("Downloading weight checkpoint for SAM")
            self._download_weight(SAM_CHECKPOINT_PATH, SAM_CHECKPOINT_URL)
  
        ## Loading model
        self.device = device
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
        self.sam_predictor = SamPredictor(sam)

        self.box_annotator = sv.BoxAnnotator(thickness=1, text_scale=0.4, text_padding=7)
        self.mask_annotator = sv.MaskAnnotator()


    def _segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    

    def _download_weight(self, path, url):
        os.makedirs(weight_dir, exist_ok=True)
        os.chdir(weight_dir)
        
        subprocess.run(["wget", "-p", "–show-progress", "-O", path, url])
    
    def generate_masked_images(self, query_image, caption, save_file_path, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD):
        image = cv2.imread(query_image)

        detections = self.grounding_dino_model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        labels = detections[1]
        detections = detections[0]
        
        detections.mask = self._segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        annotated_image = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        cv2.imwrite(save_file_path, annotated_image)
        return annotated_image

    @lru_cache(maxsize=None)
    def process(self, file_path: tuple, save_name = None):
        return None

    def end(self):
        """Save state in save_path."""
        return None

    def save_state(self, save_path):
        return None


if __name__ == "__main__":
    engine = GoundingSAMEngine()
