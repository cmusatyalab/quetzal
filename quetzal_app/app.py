from quetzal_app.page_video_comparison import video_result
from quetzal_app.page_file_system import file_system

import streamlit as st
import base64

from copy import deepcopy

from streamlit_image_comparison import image_comparison

# from streamlit_javascript import st_javascript

from streamlit_elements import elements, mui, lazy, html, event
from quetzal_app.mui_components import *
import time
import datetime
from streamlit_extras.stylable_container import stylable_container

# import streamlit_shadcn_ui as ui
# import streamlit_antd_components as sac

from streamlit_tags import st_tags
from streamlit_float import *

# from quetzal_app.external.streamlit_float import *

from quetzal.video import *
from quetzal.align_frames import align_video_frames, align_frame_pairs
from quetzal_app.utils.utils import *
from quetzal_app.dtos import *
from quetzal.engines.detection_engine.grounding_sam_engine import GoundingSAMEngine

import argparse
import torch
from glob import glob
import os

import pickle
from quetzal_app.image_frame_component import image_frame
from streamlit.components.v1 import html as html_st, iframe
from streamlit_js_eval import (
    streamlit_js_eval,
    copy_to_clipboard,
    create_share_link,
    get_geolocation,
)

from collections import defaultdict
import logging
from threading import Lock

TEMP_MATCH_FILENAME = "./match_obj.pkl"
BENCH_MARK = False
BORDER_RADIUS = "0.8rem"
DEFAULT_BOX_TH = 0.3
DEFAULT_TEXT_TH = 0.25
DEFAULT_SLIDER_VAL = 0

dataset_layout_help = """
        Dataset structure:
        root_datasets_dir/
        |
        ├── project_name/
        |   ├── raw_videos/
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

st.set_page_config(layout="wide")

@st.cache_data
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="This program aligns Two Videos",
        epilog=dataset_layout_help,
    )

    parser.add_argument(
        "--dataset-root", default="../../data/root", help="Root directory of datasets"
    )
    parser.add_argument(
        "--metadata-root", default="../../data/meta_data", help="Meta data directory of datasets"
    )
    parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=True)
    parser.add_argument("--cuda_device", help="Select cuda device", default=0, type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    meta_data_root = args.metadata_root

    available_gpus = torch.cuda.device_count()
    print(f"Avaliable GPU={available_gpus}")
    if args.cuda and available_gpus > 0:
        cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
        torch_device = torch.device("cuda:" + str(cuda_device))
    else:
        torch_device = torch.device("cpu")

    print(torch_device)

    return dataset_root, meta_data_root, cuda_device, torch_device

dataset_root, meta_data_root, cuda_device, torch_device = parse_args()
user = "jinho"

user_default = {
    "root_dir":os.path.join(dataset_root, user), 
    "metadata_dir":os.path.join(meta_data_root, user), 
    "user":user, 
    "path":"./", 
    "mode":"user",
}

shared_default = {
    "root_dir":dataset_root, 
    "metadata_dir":meta_data_root, 
    "user":user, 
    "path":"./", 
    "mode":"shared",
}

if "page_state" not in st.session_state:
    curr_dir = QuetzalFile(**user_default)
    st.session_state.page_state = {
        "curr_dir": curr_dir, 
        "info_file": None,
        "compare": {"project": None, "database": None, "query": None},
        "user": user,
        "last_dir": curr_dir,
        "menu": "user",
        "info": None,
        # object Detection
        "playback": {"slider": DEFAULT_SLIDER_VAL},
        "object": {
            "slider_box_th": DEFAULT_BOX_TH,
            "slider_text_th": DEFAULT_TEXT_TH,
            "class_prompts": ["objects"],
        },
        
        
        "init": [dataset_root, meta_data_root, cuda_device, torch_device],
        "page": "file_system"
    }
    st.session_state.lock = Lock()
    
if st.session_state.page_state["page"] == "file_system":
    file_system(user)
elif st.session_state.page_state["page"] == "video":
    video_result(
        **st.session_state.page_state["compare"]
    )