from quetzal.dtos.video import *
import logging
from quetzal.engines.vpr_engine.anyloc_engine import AnyLocEngine
import torch
import sys

# Compute_vlad - For "Detection of Tunable and Explainable Salient Changes".
logging.basicConfig()
logger = logging.getLogger("compute_vlad")
logger.setLevel(logging.DEBUG)

import argparse

dataset_layout_help = """
Your dataset directory will be structured as following
Place your desired video files in dataset_root/route_name/raw_videos/
    
    Dataset structure:
    dataset_root/
    |
    ├── route_name/
    |   ├── raw_video/
    |   |   ├── video_name.mp4
    |   |   └── ...
    |   |
    |   ├── database/
    |   |   ├── video_name/
    |   |   |   ├── frames_{fps}_{resolution}/
    |   |   |   |   ├── frame_%05d.jpg
    |   |   |   |   └── ...
    |   |   |   └── ...
    |   |   └── ...
    |   |
    |   ├── query/
    |   |   ├── video_name/
    |   |   |   ├── frames_{fps}_{resolution}/
    |   |   |   |   ├── frame_%05d.jpg
    |   |   |   |   └── ...
    |   |   |   └── ...
    |   |   └── ...
    |   └── ...
    └── ...
    """

def generate_VLAD(database_video: Video, query_video: Video, torch_device):
    """
    Generates VLAD descriptors for the given database and query videos.

    Args:
        database_video (Video): The video object representing the database video.
        query_video (Video): The video object representing the query video.
        torch_device (torch.device): The PyTorch device to use for computations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the VLAD descriptors for the database and query videos.
    """
    
    logger.info("Loading Videos")
    anylocEngine = AnyLocEngine(
        database_video=database_video,
        query_video=query_video,
        device=torch_device,
        mode="lazy",
    )

    db_vlad = anylocEngine.get_database_vlad()
    query_vlad = anylocEngine.get_query_vlad()
    del anylocEngine

    return db_vlad, query_vlad


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="This program computes  ",
        epilog=dataset_layout_help,
    )
    # ... add arguments to parser ...
    parser.add_argument(
        "--dataset-root", default="../data", help="Root directory of datasets"
    )
    parser.add_argument("--route-name", required=True, help="Name of the route")
    parser.add_argument("--database-video", help="Database video file name")
    parser.add_argument("--query-video", help="Query video file name")
    parser.add_argument(
        "--cuda", action="store_true", help="Enable cuda", default=False
    )
    parser.add_argument("--cuda_device", help="Select cuda device", default=0, type=int)

    args = parser.parse_args()

    if not (args.database_video or args.query_video):
        parser.print_usage()
        print("Error: Either --database-video or --query-video must be provided.")
        sys.exit(1)

    device = torch.device("cpu")
    available_gpus = torch.cuda.device_count()
    if args.cuda and available_gpus > 0:
        cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
        device = torch.device("cuda:" + str(cuda_device))

    ## Initialize System

    # Load Video frames
    logger.info("Loading Videos")
    database_video, query_video = None, None

    if args.database_video:
        database_video = DatabaseVideo(
            datasets_dir=args.dataset_root,
            route_name=args.route_name,
            video_name=args.database_video,
        )
    if args.query_video:
        query_video = QueryVideo(
            datasets_dir=args.dataset_root,
            route_name=args.route_name,
            video_name=args.query_video,
        )

    generate_VLAD(database_video, query_video, device)


if __name__ == "__main__":
    main()
