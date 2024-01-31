from quetzal.video import *
import logging
from quetzal.engines.image_registration_engine.ransac_flow_engine import RansacFlowEngine
from quetzal.engines.vpr_engine.anyloc_engine import AnyLocEngine
from quetzal.engines.image_registration_engine.loftr_engine import LoFTREngine
from quetzal.utils.dtw_vlad import *
import torch.nn.functional as F

import torch
import sys

# generate_aligned_images
logging.basicConfig()
logger = logging.getLogger("generate_aligned_images")
logger.setLevel(logging.DEBUG)
from stqdm import stqdm

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


# def _run_alignment(
#     dataset_root: QuetzalFile, 
#     project: QuetzalFile, 
#     database: QuetzalFile, 
#     query: QuetzalFile, 
#     overlay
# ):
#     ## Load DTW and VLAD Features ##
#     database_video = DatabaseVideo(
#         datasets_dir=os.path.abspath(database.root_dir),
#         project_name=os.path.dirname(database._path),
#         video_name=os.path.basename(database._path),
#         metadata_dir=os.path.abspath(database.metadata_dir)
#     )
#     query_video = QueryVideo(
#         datasets_dir=os.path.abspath(query.root_dir),
#         project_name=os.path.dirname(query._path),
#         video_name=os.path.basename(query._path),
#         metadata_dir=os.path.abspath(query.metadata_dir)
#     )

#     db_frame_list = database_video.get_frames()
#     query_frame_list = query_video.get_frames()
#     overlay_query_frame_list = query_frame_list

#     if not overlay:
#         matches = align_video_frames(
#             database_video=database_video,
#             query_video=query_video,
#             torch_device=torch_device,
#         )
#     else:
#         matches, overlay_query_frame_list = align_frame_pairs(
#             database_video=database_video,
#             query_video=query_video,
#             torch_device=torch_device,
#         )

#     return {
#         "matches": matches,
#         "query_frame_list": query_frame_list,
#         "db_frame_list": db_frame_list,
#         "overlay_query_frame_list": overlay_query_frame_list,
#     }


def align_video_frames(database_video: Video, query_video: Video, torch_device):
    anylocEngine = AnyLocEngine(
        database_video=database_video, query_video=query_video, device=torch_device
    )

    db_vlad = anylocEngine.get_database_vlad()
    query_vlad = anylocEngine.get_query_vlad()

    del anylocEngine

    # Normalize and prepare x and y for FAISS
    db_vlad = F.normalize(db_vlad)
    query_vlad = F.normalize(query_vlad)
    cuda = torch_device != torch.device("cpu")
    try:
        db_indexes = create_FAISS_indexes(db_vlad.numpy(), cuda=cuda)
    except:
        db_indexes = create_FAISS_indexes(db_vlad.numpy(), cuda=False)

    ## Run DTW Algorithm using VLAD features ##
    _, _, D1, path = dtw(query_vlad.numpy(), db_vlad, db_indexes)
    matches = extract_unique_dtw_pairs(path, D1)

    # Smooth the frame alignment Results
    query_fps = query_video.get_fps()
    db_fps = database_video.get_fps()

    diff = 1
    count = 0
    k = 3
    while diff and count < 100:
        time_diff = [
            database_video.get_frame_time(d) - query_video.get_frame_time(q)
            for q, d in matches
        ]
        mv_avg = np.convolve(time_diff, np.ones(k) / k, mode="same")
        mv_avg = {k[0]: v for k, v in zip(matches, mv_avg)}
        matches, diff = smooth_frame_intervals(matches, mv_avg, query_fps, db_fps)
        count += 1
    return matches


def align_frame_pairs(database_video: Video, query_video: Video, torch_device, engine: Literal["ransac-flow", "loftr"] = "loftr"):
    logger.info("Loading Videos")

    if engine == "ransac-flow":
        engine = RansacFlowEngine(
            query_video,
            max_img_size=1024,
            fine_align=False,
            device=torch_device,
            db_name=database_video.get_video_name(),
        )
    elif engine == "loftr":
        engine = LoFTREngine(
            query_video,
            device=torch_device,
            db_name=database_video.get_video_name(),
        )

    matches = align_video_frames(database_video=database_video, 
                                 query_video=query_video,
                                 torch_device=torch_device)
    query_frame_list = query_video.get_frames()
    db_frame_list = database_video.get_frames()

    aligned_frame_list = list()
    for query_idx, db_idx in stqdm(matches, desc="Generating Overlay frames", backend=True):
        query_frame = query_frame_list[query_idx]
        db_frame = db_frame_list[db_idx]
        aligned_frame_list.append(engine.process((query_frame, db_frame))[0])

    del engine

    return matches, aligned_frame_list

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

    align_frame_pairs(database_video, query_video, device)


if __name__ == "__main__":
    main()