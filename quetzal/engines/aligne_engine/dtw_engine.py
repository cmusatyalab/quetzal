from os.path import join
import torch

import logging
from quetzal.engines.engine import AlignmentEngine, FrameMatch, WarpedFrame
from quetzal.align_frames import align_frame_pairs, align_video_frames
from quetzal.dtos.video import Video

logging.basicConfig()
logger = logging.getLogger("DTW Engine")
logger.setLevel(logging.DEBUG)

class DTWEngine(AlignmentEngine):
    """
    DTWEngine (Dynamic Time Warping Engine) is designed for frame alignment between a database video and a query video. 
    It utilizes dynamic time warping to find the best alignment between frames of the two videos, facilitating tasks 
    such as video synchronization, comparison, or analysis.
    
    Attributes:
        device (torch.device): The device on which computations are performed, typically a GPU for efficiency.
    """
    name = "grounding_sam"
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
    ):
        """
        Initializes the DTWEngine with a specified computation device.

        Args:
            device (torch.device): The device for computations, defaults to GPU ('cuda:0').
        """

        ## Loading model
        self.device = device
    
    def align_frame_list(
        self, database_video: Video, query_video: Video, overlay: bool
    ) -> tuple[FrameMatch, WarpedFrame]:
        """
        Aligns frames between a database video and a query video. Depending on the 'overlay' flag, 
        it either performs alignment to find matching frames or also generates a list of warped (transformed) 
        query frames for overlay on the database video frames.

        Args:
            database_video (Video): The database video against which the query video is aligned.
            query_video (Video): The query video to be aligned with the database video.
            overlay (bool): If True, generates warped query frames for overlay purposes. If False, only aligns frames.

        Returns:
            tuple[FrameMatch, WarpedFrame]: A tuple containing two elements:
                - FrameMatch: A list of tuple pairs indicating matched frames between database and query videos.
                - WarpedFrame: A list of warped query video frames, returned only if 'overlay' is True.
        """
        
        # db_frame_list = database_video.get_frames()
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