import os
from os.path import join
import numpy as np
import torch
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from src.external.AnyLoc.utilities import DinoV2ExtractFeatures
from src.external.AnyLoc.utilities import VLAD
from src.video import Video
from typing import Literal, List
from tqdm import tqdm
from PIL import Image
from functools import lru_cache
import logging

from src.engines.engine import AbstractEngine
from torch.nn import functional as F
import faiss

_ex = lambda x: os.path.realpath(os.path.expanduser(x))
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
anyloc_dir = join(current_file_dir, "../../external/AnyLoc")
cache_dir: str = _ex(join(anyloc_dir, "cache"))

logging.basicConfig()
logger = logging.getLogger("AnyLoc_Engine")
logger.setLevel(logging.DEBUG)

class AnyLocEngine(AbstractEngine):
    def __init__(
        self,
        database_video: Video = None,  # database_video may be None. Can later register using register_db_video()
        query_video: Video = None,  # query_video may be None. Can later register using register_query_video()
        max_img_size: int = 512,
        device=torch.device("cuda:0"),

        domain: Literal["aerial", "indoor", "urban"] = "aerial",
        mode: Literal["vpr", "realtime", "lazy"] = "realtime",
    ):
        """
        Initializes the AnyLocEngine with optional database and query videos.

        Args:
            database_video (Video, optional): The database video for analysis.
            query_video (Video, optional): The query video for analysis.
            max_img_size (int, optional): Maximum size of the images during processing.
            device (torch.device, optional): The computational device (CPU/GPU).
            domain (Literal["aerial", "indoor", "urban"], optional): The domain type of the videos.
            mode (Literal["vpr", "realtime", "lazy"], optional): The operational mode of the engine.

        Attributes:
            name (str): Name of the engine.
            db_video (Video): The database video.
            query_video (Video): The query video.
            db_index (faiss.IndexFlatIP): FAISS index for the database VLAD vectors.
            query_vlad (np.ndarray): The VLAD vectors for the query video.

        Notes:
            "vpr": This mode is optimized for retrieval of the closest database frames from a query image. It pre-computes both database and query VLAD features and prepares a database FAISS index for quick retrieval.
            "realtime": This mode is optimized for real-time frame retrieval and does not pre-computes the query VLAD features, assuming that not all of the frames are ready.  Suitable for scenarios where per frames based real-time processing is required
            "lazy": This mode is optimized for computing entire VLAD features of each Video in a blocking manner. In this mode, VLAD features for both the query and database videos are not computed during initialization. Instead, the computation is deferred until the user explicitly calls the get_vlad_features() method. Calling process() method for VPR will be disabled.
            
            The `domain` parameter in the `AnyLocEngine` class is used to specify the domaintype of the videos being processed. It is a literal type that can take one of three values: "aerial", "indoor", or "urban". This parameter is used to determine the location of the cluster centers file, which is required for VLAD feature aggregation. The cluster centers file is stored in the cache directory, and the path to the file is constructed based on the specified domain type.
        """
        self.name = "Frame Matching - AnyLoc"

        ## Video Frames ##
        self.db_video = None
        self.query_video = None

        ## Check Model Cache ##
        if os.path.isdir(cache_dir):
            logger.info("Anyloc cache folder already exists!")
        else:
            self._download_cache()

        ## DINOv2 Extractor ##
        self.max_img_size = max_img_size
        self.device = device

        vlad_ready = self._is_vlad_ready(database_video) and self._is_vlad_ready(
            query_video
        )

        if not vlad_ready:
            desc_layer: int = 31
            desc_facet: Literal["query", "key", "value", "token"] = "value"
            num_c: int = 32
            # Domain for use case (deployment environment)
            domain: Literal["aerial", "indoor", "urban"] = domain

            self.extractor = DinoV2ExtractFeatures(
                "dinov2_vitg14", desc_layer, desc_facet, device=device
            )
            self.base_tf = tvf.Compose(
                [
                    tvf.ToTensor(),
                    tvf.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        ## VLAD object ##
        if not vlad_ready:
            ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
            c_centers_file = os.path.join(
                cache_dir, "vocabulary", ext_specifier, domain, "c_centers.pt"
            )
            assert os.path.isfile(c_centers_file), "Cluster centers not cached!"
            c_centers = torch.load(c_centers_file)
            assert c_centers.shape[0] == num_c, "Wrong number of clusters!"

            self.vlad = VLAD(
                num_c, desc_dim=None, cache_dir=os.path.dirname(c_centers_file)
            )
            # Fit (load) the cluster centers (this'll also load the desc_dim)
            self.vlad.fit(None)

        ## Initialize VLAD features ##
        self.db_index = None
        self.query_vlad = None
        self.query_vlad_cache = []

        ## Register Videos ##
        self.register_db_video(database_video, mode)
        self.register_query_video(query_video, mode)

    def _is_vlad_ready(self, video: Video) -> bool:
        """
        Checks if the VLAD features are already computed for a given video.

        Args:
            video (Video): The video object to check for precomputed VLAD features.

        Returns:
            bool: True if the VLAD features are precomputed, False otherwise.
        """
        return (
            os.path.isfile(f"{video.get_dataset_dir()}/vlads.npy") if video else False
        )

    def register_db_video(
        self, database_video: Video, mode: Literal["vpr", "realtime", "lazy"] = "vpr"
    ):
        """
        Registers a database video in the engine and initializes its VLAD features.

        Args:
            database_video (Video): The video to be registered as a database video.
            mode (Literal["vpr", "realtime", "lazy"], optional): The mode of operation for VLAD feature computation.

        Note:
            This method initializes the FAISS index for the database VLAD vectors if the mode is "vpr" or "realtime".
        """

        if self.db_video:
            logger.info("Database Video Already Registered")
            return

        self.db_video = database_video
        self.db_img_frames = (
            database_video.get_frames(verbose=False) if database_video else []
        )

        ## Initialize Database VLAD Features ##
        if database_video and mode in ["vpr", "realtime"]:
            db_vlad = self._get_vlad_set(self.db_video)
            db_vlad = F.normalize(db_vlad)
            D = db_vlad.shape[1]
            self.db_index = faiss.IndexFlatIP(D)
            res = faiss.StandardGpuResources()
            self.db_index = faiss.index_cpu_to_gpu(res, 0, self.db_index)
            self.db_index.add(db_vlad.numpy())

    def register_query_video(
        self, query_video: Video, mode: Literal["vpr", "realtime", "lazy"] = "vpr"
    ):
        """
        Registers a query video in the engine.

        Args:
            query_video (Video): The video to be registered as a query video.
            mode (Literal["vpr", "realtime", "lazy"], optional): The mode of operation for VLAD feature computation.

        Note:
            This method does not immediately compute VLAD features for the query video unless the mode is "vpr".
        """

        if self.query_video:
            logger.info("Query Video Already Registered")
            return

        self.query_video = query_video
        self.query_img_frames = (
            query_video.get_frames(verbose=False) if query_video else []
        )

        ## Initialize Query VLAD Features ##
        if query_video:
            if mode == "vpr":
                self.query_vlad = self._get_vlad_set(self.query_video)
            elif mode == "realtime":
                ## Load Cached query_vlad features if exist ##
                if query_video and os.path.isfile(
                    f"{query_video.get_dataset_dir()}/vlads.npy"
                ):
                    self.query_vlad = np.load(
                        f"{query_video.get_dataset_dir()}/vlads.npy"
                    )
                else:
                    self.query_vlad = None

    def get_query_vlad(self) -> np.ndarray:
        """
        Retrieves the VLAD features for the registered query video.

        Returns:
            np.ndarray: The VLAD features of the query video.
        """
        return self._get_vlad_set(self.query_video) if self.query_video else None

    def get_database_vlad(self) -> np.ndarray:
        """
        Retrieves the VLAD features for the registered database video.

        Returns:
            np.ndarray: The VLAD features of the database video.
        """
        return self._get_vlad_set(self.db_video) if self.db_video else None

    def _get_vlad_set(self, video: Video) -> np.ndarray:
        """
        Computes and retrieves VLAD features for a given video.

        Args:
            video (Video): The video for which to compute VLAD features.

        Returns:
            np.ndarray: The computed VLAD features for the video.
        """
        dataset_folder = video.get_dataset_dir()
        max_img_size = self.max_img_size

        if not os.path.isfile(f"{dataset_folder}/vlads.npy"):
            logger.info(f"Generating VLAD features for the Video {video.video_name}")

            patch_descs = []
            img_frames = video.get_frames(verbose=False)

            for img_fname in tqdm(img_frames):
                # DINO features
                with torch.no_grad():
                    pil_img = Image.open(img_fname).convert("RGB")
                    img_pt = self.base_tf(pil_img).to(self.device)
                    if max(img_pt.shape[-2:]) >= max_img_size:
                        c, h, w = img_pt.shape
                        # Maintain aspect ratio
                        if h == max(img_pt.shape[-2:]):
                            w = int(w * max_img_size / h)
                            h = max_img_size
                        else:
                            h = int(h * max_img_size / w)
                            w = max_img_size
                        img_pt = T.resize(
                            img_pt, (h, w), interpolation=T.InterpolationMode.BICUBIC
                        )
                    # Make image patchable (14, 14 patches)
                    c, h, w = img_pt.shape
                    h_new, w_new = (h // 14) * 14, (w // 14) * 14
                    img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
                    ret = self.extractor(img_pt)  # [1, num_patches, desc_dim]
                    patch_descs.append(ret.cpu())

            with torch.no_grad():
                patch_descs = torch.cat(patch_descs, dim=0)
                img_names = [None] * len(img_frames)
                vlads: torch.Tensor = self.vlad.generate_multi(patch_descs, img_names)
            del patch_descs
            np.save(f"{dataset_folder}/vlads.npy", vlads)

        else:
            vlads = np.load(f"{dataset_folder}/vlads.npy")
            vlads = torch.from_numpy(vlads)

        return vlads

    def _get_vlad(self, frame: str) -> np.ndarray:
        """
        Computes the VLAD feature for a single frame.

        Args:
            frame (str): The path to the frame image file.

        Returns:
            np.ndarray: The computed VLAD feature for the frame.
        """
        max_img_size = self.max_img_size

        # DINO features
        with torch.no_grad():
            pil_img = Image.open(frame).convert("RGB")
            img_pt = self.base_tf(pil_img).to(self.device)
            if max(img_pt.shape[-2:]) > self.max_img_size:
                c, h, w = img_pt.shape
                # Maintain aspect ratio
                if h == max(img_pt.shape[-2:]):
                    w = int(w * max_img_size / h)
                    h = max_img_size
                else:
                    h = int(h * max_img_size / w)
                    w = max_img_size
                img_pt = T.resize(
                    img_pt, (h, w), interpolation=T.InterpolationMode.BICUBIC
                )
            # Make image patchable (14, 14 patches)
            c, h, w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
            # Extract descriptor
            ret = self.extractor(img_pt)  # [1, num_patches, desc_dim]
        # VLAD global descriptor
        gd = self.vlad.generate(ret.cpu().squeeze())  # VLAD: shape [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...]  # shape: [1, agg_dim]

        return gd_np

    def _download_cache(self):
        """
        Downloads and sets up the cache folder necessary for the AnyLoc engine, including DINOv2 model and VLAD cluster centers.
        """
        from external.AnyLoc.utilities import od_down_links
        from onedrivedownloader import download

        # Link
        ln = od_down_links["cache"]

        # Download and unzip
        logger.info("Downloading the cache folder")
        download(
            ln, filename="cache.zip", unzip=True, unzip_path=_ex(anyloc_dir), clean=True
        )
        logger.info("Cache folder downloaded")

    @lru_cache(maxsize=None)
    def process(self, file_path: str):
        """
        Processes a given file to find its matching frame in the database video.

        Args:
            file_path (str): The path to the query file.

        Returns:
            Tuple[str, str]: A tuple containing the file path and the path to its matching frame in the database.
        """
        if self.query_vlad is not None:
            idx = self.query_video.get_frame_idx(file_path)
            query = self.query_vlad[idx]
            query = query[np.newaxis, ...]

        else:
            query = self._get_vlad(file_path)
            self.query_vlad_cache.append(query)

        _distances, indices = self.db_index.search(query, max([1]))
        match_image = self.db_img_frames[indices[0][0]]

        return (file_path, match_image)

    def end(self):
        """
        Concludes the processing and performs necessary cleanup.
        """
        self.save_state()
        return None

    def save_state(self, save_path: str):
        """
        Saves the current state of the engine, including cached VLAD features.

        Args:
            save_path (str): The path where the state should be saved.
        """
        self._save_query_vlad()

    def _save_query_vlad(self):
        if self.query_video.get_frame_len() == len(self.query_vlad_cache):
            dataset_folder = self.query_video.get_dataset_dir()
            query_vlad = np.concatenate(self.query_vlad_cache, axis=0)
            np.save(f"{dataset_folder}/vlads.npy", query_vlad)
            self.query_vlad = query_vlad


if __name__ == "__main__":
    engine = AnyLocEngine()
