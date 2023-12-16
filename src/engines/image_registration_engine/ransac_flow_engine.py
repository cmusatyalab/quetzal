import os
from os.path import join
from glob import glob
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from src.external.RANSAC_flow.utils import outil
from src.external.RANSAC_flow.coarseAlignFeatMatch import CoarseAlign
import src.external.RANSAC_flow.model.model as ransac_model
from tqdm import tqdm
from PIL import Image
import kornia.geometry as tgm

from functools import lru_cache
import logging
from src.video import Video
from src.engines.engine import AbstractEngine


logging.basicConfig()
logger = logging.getLogger("RASNAC_flow_Engine")
logger.setLevel(logging.DEBUG)

_ex = lambda x: os.path.realpath(os.path.expanduser(x))
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
ransac_flow_dir = join(current_file_dir, "../../external/RANSAC_flow")
resumePth: str = _ex(
    join(
        ransac_flow_dir,
        "model",
        "pretrained",
        "MegaDepth_Theta1_Eta001_Grad1_0.774.pth",
    )
)


class RansacFlowEngine(AbstractEngine):
    def __init__(
        self,
        query_video: Video,
        max_img_size: int = 1920,
        fine_align=False,
        device: torch.device=torch.device("cuda:0"),
        db_name: str="",
        nb_scale = 7,
        coarse_iter = 30000,
        coarse_tolerance = 0.03,
        scale_r = 1.2
    ):
        """
        Assumes Using GPU (cuda)
        """

        self.name = "Image Alignment - RANSAC-flow"

        ## Save directory
        if fine_align:
            align_mode = "f"
        else:
            align_mode = "c"
        self.save_dir = join(
            query_video.get_dataset_dir(),
            db_name + "_ransac_flow_" + align_mode + str(max_img_size) + "_" + str(coarse_iter),
        )
        os.makedirs(self.save_dir, exist_ok=True)

        ## Loading model
        kernelSize = 7
        self.device = device

        if not os.path.isfile(resumePth):
            logger.info("Downloading pretrained models")
            self._download_pretrained()

        # Define Networks
        self.network = {
            "netFeatCoarse": ransac_model.FeatureExtractor(),
            "netCorr": ransac_model.CorrNeigh(kernelSize),
            "netFlowCoarse": ransac_model.NetFlowCoarse(kernelSize),
            "netMatch": ransac_model.NetMatchability(kernelSize),
        }

        for key in list(self.network.keys()):
            self.network[key].to(device=device)
            # self.network
            # typeData = torch.cuda.FloatTensor

        # loading Network
        param = torch.load(resumePth)
        msg = "Loading pretrained model from {}".format(resumePth)
        logger.info(msg)

        for key in list(param.keys()):
            self.network[key].load_state_dict(param[key])
            self.network[key].eval()


        imageNet = True  # we can also use MOCO feature here
        self.coarseModel = CoarseAlign(
            nb_scale,
            coarse_iter,
            coarse_tolerance,
            "Homography",
            max_img_size,
            1,
            True,
            imageNet,
            scale_r,
            device=self.device,
        )

    def _download_pretrained(self):
        import requests
        from zipfile import ZipFile

        # Set the expected number of models
        EXPECT_NUM_MODELS = 8
        target_directory = os.path.pardir(resumePth)

        # Create the target directory if it doesn't exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # List all .pth files in the target directory
        models = [f for f in os.listdir(target_directory) if f.endswith(".pth")]
        NUM_DOWNLOADED_MODELS = len(models)

        # Check if the number of models is less than expected
        if NUM_DOWNLOADED_MODELS < EXPECT_NUM_MODELS:
            # Define the URL of the zip file
            url = "https://www.dropbox.com/s/uegv8aqq5sj3542/model.zip?dl=1"  # dl=1 for direct download
            # Define the local filename to save the downloaded file
            local_filename = os.path.join(target_directory, "model.zip")

            # Stream download the file to avoid loading it all into memory at once
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Unzip the file after download
            with ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(target_directory)

            # Remove the zip file after extraction
            os.remove(local_filename)

    def _generate_aligned_images(
        self, query_image, database_image, save_file_path, fine_align=False
    ):
        I1 = Image.open(query_image).convert("RGB")
        I2 = Image.open(database_image).convert("RGB")

        self.coarseModel.setSource(I1)
        self.coarseModel.setTarget(I2)

        I2w, I2h = self.coarseModel.It.size
        featt = F.normalize(self.network["netFeatCoarse"](self.coarseModel.ItTensor))

        #### -- grid
        gridY = (
            torch.linspace(-1, 1, steps=I2h).view(1, -1, 1, 1).expand(1, I2h, I2w, 1)
        )
        gridX = (
            torch.linspace(-1, 1, steps=I2w).view(1, 1, -1, 1).expand(1, I2h, I2w, 1)
        )
        warper = tgm.HomographyWarper(I2h, I2w)

        bestPara, InlierMask = self.coarseModel.getCoarse(np.zeros((I2h, I2w)))
        bestPara = torch.from_numpy(bestPara).unsqueeze(0).to(device=self.device)

        I1_coarse = warper(self.coarseModel.IsTensor, bestPara)

        if not fine_align:
            I1_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())
        else:
           raise("No longer supporting Fine grain alignment")

        # Save the image
        I1_pil.save(save_file_path)

    @lru_cache(maxsize=None)
    def process(self, file_path: tuple):
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

    def save_state(self, save_path):
        return None


if __name__ == "__main__":
    engine = RansacFlowEngine()
