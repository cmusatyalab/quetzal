import os
from os.path import join
import torch
import subprocess
from groundingdino.util.inference import Model
import supervision as sv
from streamlit_label_kit import absolute_to_relative, relative_to_absolute

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from segment_anything import SamPredictor

import cv2

import logging
from quetzal.engines.engine import ObjectDetectionEngine

logging.basicConfig()
logger = logging.getLogger("GroundingSAM Engine")
logger.setLevel(logging.DEBUG)

_ex = lambda x: os.path.realpath(os.path.expanduser(x))
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
grounding_path = join(current_file_dir, "../../external/GroundingDINO")
weight_dir = join(current_file_dir, "../../../weights")

GROUNDING_DINO_CHECKPOINT_PATH: str = _ex(
    join(weight_dir, "groundingdino_swint_ogc.pth")
)
GROUNDING_DINO_CONFIG_PATH: str = _ex(
    join(grounding_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
)

BOX_TRESHOLD = 0.25  # 0.3
TEXT_TRESHOLD = 0.25

SAM_CHECKPOINT_PATH = os.path.join(weight_dir, "sam_vit_h_4b8939.pth")
SAM_ENCODER_VERSION = "vit_h"

SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
GROUNDING_DINO_CHECKPOINT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"


class GroundingSAMEngine(ObjectDetectionEngine):
    """
    GroundingSAMEngine integrates both GroundingDINO and SAM (Segment Anything Model) to perform object detection
    and segmentation based on textual descriptions. This engine loads pretrained weights for GroundingDINO and SAM,
    applies text-based object detection using GroundingDINO, and leverages SAM for accurate segmentation of detected objects.

    Attributes:
        device (torch.device): The device on which the models will be loaded and inference will be performed.
        save_dir (str): Directory where temporary files or results may be saved.

    Methods:
        generate_masked_images: Processes an input image using a textual description to detect and segment objects, 
                                saving the resulting image with annotations.
    """
    name = "grounding_sam"
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
    ):
        """
        Initializes the GroundingSAMEngine with the specified device for computation.

        Args:
            device (torch.device): Specifies the device (e.g., CPU, GPU) for model computation. Defaults to GPU.
        """

        self.save_dir = join(current_file_dir, "../../../tmp")
        os.makedirs(self.save_dir, exist_ok=True)

        if not os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH):
            logger.info("Downloading weight checkpoint for GroundingDINO")
            self._download_weight(
                GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CHECKPOINT_URL
            )

        if not os.path.isfile(SAM_CHECKPOINT_PATH):
            logger.info("Downloading weight checkpoint for SAM")
            self._download_weight(SAM_CHECKPOINT_PATH, SAM_CHECKPOINT_URL)

        ## Loading model
        self.device = device
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
        sam = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH
        ).to(device=device)
        self.sam_predictor = SamPredictor(sam)

        self.box_annotator = sv.BoxAnnotator(
            thickness=1, text_scale=0.4, text_padding=7
        )
        self.mask_annotator = sv.MaskAnnotator()

    def _segment(
        self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
    ) -> np.ndarray:
        """
        Uses SAM to segment the specified regions in an image.

        Args:
            sam_predictor (SamPredictor): The SAM predictor for performing segmentation.
            image (np.ndarray): The input image as a NumPy array.
            xyxy (np.ndarray): An array of bounding boxes, where each box is defined by [x1, y1, x2, y2].

        Returns:
            np.ndarray: An array of segmented masks corresponding to the input bounding boxes.
        """
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def _download_weight(self, path, url):
        """
        Downloads a model weight file from a specified URL to a given path.

        Args:
            path (str): The local file path where the weight should be saved.
            url (str): The URL from which the weight file will be downloaded.
        """
        os.makedirs(weight_dir, exist_ok=True)
        os.chdir(weight_dir)

        subprocess.run(["wget", "-p", "–show-progress", "-O", path, url])

    def generate_masked_images(
        self,
        query_image: str,
        caption: list[str],
        save_file_path: str,
        box_threshold: float = BOX_TRESHOLD,
        text_threshold: float = TEXT_TRESHOLD,
        show_background: bool = True
    ):
        """
        Generates an image with detected objects based on the provided captions masked and annotated. 
        Detected objects are based on textual descriptions and segmented using SAM.

        Args:
            query_image (str): Path to the input image.
            caption (list[str]): A list of captions (textual descriptions) for object detection.
            save_file_path (str): Path where the annotated image will be saved.
            box_threshold (float, optional): Threshold for object detection bounding boxes. Defaults to BOX_TRESHOLD.
            text_threshold (float, optional): Threshold for textual descriptions. Defaults to TEXT_TRESHOLD.
            show_background (bool, optional): Show Segmentation and detections on background. Defaults to True for object detection tab
        
        Returns:
            np.ndarray: The annotated image with detected and segmented objects based on captions.
            np.ndarray: A list of the xyxy positions of objects based on captions.
            np.ndarray: A list of labels corresponding to the detected objects.
        """
        caption = " . ".join(caption)

        image = cv2.imread(query_image)

        height, width, _ = image.shape

        detections = self.grounding_dino_model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        labels = detections[1]
        detections = detections[0]

        detections.mask = self._segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        if show_background:
            # annotate image with detections
            annotated_image = self.mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )
        else:
            annotated_image = image.copy()

        cv2.imwrite(save_file_path, annotated_image)

        xyxy_relative = [absolute_to_relative(bbox, width, height) for bbox in detections.xyxy]

        return annotated_image, xyxy_relative, labels
    def generate_segmented_images(self, query_image: str, save_file_path: str, xyxy: np.ndarray):
        """
        Generates segmented image given bounding boxes

        Args:
            query_image (str): Path to the input image.
            save_file_path (str): Path where the annotated image will be saved.
            xyxy (np.ndarray): Array of bounding boxes in relative format (xyxy)
        
        Returns:
            np.ndarray: The annotated image with detected and segmented objects based on captions.
            np.ndarray: Segment Mask
        """
        image = cv2.imread(query_image)
        height, width, _ = image.shape


        mask = self._segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=np.array([relative_to_absolute(v, width, height) for v in xyxy])
        )
        
        mask = np.logical_or.reduce(mask).astype(int)
        mask_image = (mask * 255).astype(np.uint8) 

        cv2.imwrite(save_file_path, mask_image)
        color_mask = np.zeros_like(image)
        color_mask[mask > 0.5] = [255, 255, 255] # Choose any color you like
        masked_image = cv2.addWeighted(image, 0.4, color_mask, 0.6, 0)
        cv2.imwrite(save_file_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        return image, mask
    def save_segmented_masks(self, query_mask: str, db_mask: str, save_file_path: str):
        """
        Generates a mask image for change detection dataset

        Args:
            query_mask (nd.array): Segmented query mask.
            db_mask (nd.array): Segmented database mask.
            save_file_path (str): Path where the annotated image will be saved.
        
        Returns:
            np.ndarray: Dataset image
        """

        combine_mask = np.logical_or(query_mask, db_mask).astype(np.uint8)
        mask_image = (combine_mask * 255).astype(np.uint8)
        cv2.imwrite(save_file_path, mask_image)

        return mask_image



    
if __name__ == "__main__":
    engine = GroundingSAMEngine()


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