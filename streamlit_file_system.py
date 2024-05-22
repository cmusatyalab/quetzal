import streamlit as st
import base64

from copy import deepcopy

from streamlit_image_comparison import image_comparison

# from streamlit_javascript import st_javascript

from streamlit_elements import elements, mui, lazy, html, event
from quetzal_app.elements.mui_components import *
import time
import datetime
from streamlit_extras.stylable_container import stylable_container

# import streamlit_shadcn_ui as ui
# import streamlit_antd_components as sac

from streamlit_tags import st_tags
from streamlit_float import *

# from quetzal_app.external.streamlit_float import *

from quetzal.dtos.video import *
from quetzal.align_frames import align_video_frames, align_frame_pairs
from quetzal_app.utils.utils import *
from quetzal.dtos.dtos import *
from quetzal.engines.detection_engine.grounding_sam_engine import GroundingSAMEngine

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

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


BORDER_RADIUS = "0.8rem"
BACKGROUND_COLOR = "#f8fafd"
PRIMARY_COLOR = "#c9e6fd"
GOOGLE_RED = "#EA4335"
GOOGLE_DARK_RED = "#d33a2e"
GOOGLE_DARK_BLUE = "#1266F1"
GOOGLE_BLUE = "#4285F4"
GOOGLE_LIGHT_BLUE = "#edf1f9"


top_padding = "1rem"

st.set_page_config(layout="wide")
st.markdown(
    f"""
        <style>
                # @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap'); 

                # html, body, [class*="css"] {{
                #     font-family: 'Roboto', sans-serif;
                # }}
                .block-container {{ /* Removes streamlit default white spaces in the main window*/
                    padding-top: {top_padding};
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                    background: {BACKGROUND_COLOR};
                }}
                .stSlider {{ /* Removes white spaces in the main window*/
                    padding-left: 1rem;
                    padding-right: 1rem;
                }}
                [class^="st-emotion-cache-"] {{ /* Removes streamlit default gap between containers, elements*/
                    gap: 0rem;
                    border-radius: {BORDER_RADIUS};
                    background-color: transparent
                }}
        </style>
        """,
    unsafe_allow_html=True,
)


# with open( "./style.css" ) as css:
#     st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
# float_init()

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


# Initialize Variable
dataset_root, meta_data_root, cuda_device, torch_device = parse_args()
project_name = "example_mil19"
query_video = "P0200020.MP4"
database_video = "P0190019.MP4"
grounding_sam = None

window_height = streamlit_js_eval(
    js_expressions="screen.height", want_output=True, key="WIND_H"
)
padding = "11.5rem"

MENU_WIDTH = 250
INFO_WIDTH = 340
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


#     path=EXAMPLE_LINK,
#     type=FileType.DIRECTORY,
#     visibility=Visibility.SHARED,
#     analysis_progress=AnalysisProgress.NONE,
#     permission=Permission.FULL_WRITE,
#     owner="jinhoy",
# )

# selected_file = convert_to_quetzal_files(TEST_FILE_LIST[1])

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
    }
    st.session_state.lock = Lock()

curr_dir = st.session_state.page_state["curr_dir"]
if curr_dir != st.session_state.page_state["last_dir"]:
    st.session_state.page_state["info_file"] = None
    
info_file = st.session_state.page_state["info_file"]
st.session_state.page_state["last_dir"] = curr_dir

def resetInfoFile():
    with st.session_state.lock:
        st.session_state.page_state["info_file"] = None

on_focus_handler = MuiOnFocusHandler()
# on_focus_handler.registerHandler(resetInfoFile)
file_list = None

with st.session_state.lock:
    menu_c, files_c, info_c, gap= st.columns([2, 100, 2, 1])
    with menu_c:
        with stylable_container(
            key="menu_container",
            css_styles=f"""{{
                    display: block;
                    div {{
                            width: {MENU_WIDTH}px;
                            height: auto;
                        }}
                    iframe {{
                        width: {MENU_WIDTH}px;
                        height: calc({window_height}px - {padding});
                    }}
                    # width: {MENU_WIDTH}px;
                }}
                """,
        ):
            with elements("menu"):
                with mui.Paper(
                    variant="outlined",
                    sx={
                        "borderRadius": "0px",
                        "border": "0px",
                        "width": "100%",
                        "height": f"calc({window_height}px - {padding})",
                        "bgcolor": BACKGROUND_COLOR,
                        "position": "absolute",
                        "left": "0px",
                        "top": "0px",
                    },
                ):
                    ## Title
                    mui.Typography(
                        "Quetzal",
                        sx={"fontSize": "h4.fontSize", "mx": "1rem"},
                    )

                    ## Action Menu
                    upload_menu = MuiActionMenu(
                        mode=["upload"],
                        key="upload_action_menu",
                        onClick=FileActionDialog.buildDialogOpener("main_dialog"),
                    ).render()
                    # upload_menu.render()

                    ## Upload Button
                    MuiUploadButton(
                        key="upload_button", onClick=upload_menu.buildMenuOpener(curr_dir)
                    ).render()

                    ## Side bar Menu
                    def onChangeHandler(event):
                        logger.debug("onChangeHandler: Side bar Menu")
                        with st.session_state.lock:
                            value = getEventValue(event)
                            st.session_state.page_state["menu"] = value
                            if value == "user":
                                curr_dir = QuetzalFile(**user_default)
                            elif value == "shared":
                                curr_dir = QuetzalFile(**shared_default)
                            st.session_state.page_state["curr_dir"] = curr_dir
                        
                    toggle_buttons = [
                        MuiToggleButton("user", "Home", "My Projects"),
                        MuiToggleButton("shared", "FolderShared", "Shared by Others"),
                    ]
                    MuiSideBarMenu(toggle_buttons=toggle_buttons, key="main_menu", onChange=onChangeHandler).render()
                    
                    ## Compare Prompt
                    MuiComparePrompt(
                        **st.session_state.page_state["compare"],
                        onClick=None,
                    ).render()
                    
                    on_focus_handler.setScanner(key="menu_col")
                    on_focus_handler.registerHandler(keys="menu_col", handler=lambda: MuiActionMenu.resetAnchor(
                        exculde_keys=["upload_action_menu"]
                    ))

    def draw_background(px, bp):
        mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "0",
                "border": "0px",
                "width": "100%",
                "height": f"calc({window_height}px - {padding} + 1rem)",
                "bgcolor": BACKGROUND_COLOR,
                "m": "0px",
                "p": "0px",
                "position": "absolute",
                "left": "0px",
                "top": "0px",
                "zIndex": -2,
            },
        )
        mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "1rem",
                "border": "0px",
                "width": f"calc(100% - {px}rem)",
                "height": f"calc({window_height}px - {padding} - {bp}rem)",
                "bgcolor": "white",
                "m": "0px",
                "p": "0px",
                "position": "absolute",
                "left": "0px",
                "top": "0px",
                "zIndex": -1,
            },
        )

    with files_c:
        with stylable_container(
            key="slider",
            css_styles=f"""{{
                display: block;
                div {{
                    width: 100%;
                    height: auto;
                }}
                iframe {{
                    width: calc(100%) !important;
                    height: calc({window_height}px - {padding});
                }}
            }}
            """,
        ):
            buttom_padding = 0.5
            inner_padding = 0.5
            padding_right = 0 # additional padding
            default_margin = 0.5
            with stylable_container(
                key="file2",
                css_styles=f"""{{
                    display: block;
                    border-radius: 0;
                    height: calc({window_height}px - {padding} - {buttom_padding}rem + 1rem);
                    position: absolute;
                    right: calc({INFO_WIDTH}px - 2%) !important;
                    width: calc(104% - {INFO_WIDTH}px - {MENU_WIDTH}px) !important;
                    background-color: {BACKGROUND_COLOR};
                }}
                """,
            ):
                with elements("files"):
                    # Set Background
                    draw_background(px=(inner_padding * 2 + padding_right), bp=buttom_padding)
                    with mui.Paper(
                        variant="outlined",
                        sx={
                            "borderRadius": "1rem",
                            "border": "0px",
                            "width": f"calc(100% - {inner_padding * 2}rem - {padding_right}rem - {default_margin * 2}rem)",
                            "height": f"calc({window_height}px - {padding} - {top_padding} - {inner_padding * 2}rem);",
                            "bgcolor": "white",
                            "padding": f"{inner_padding}rem",
                            "zIndex": 2,
                        },
                    ):
                        info_shown = False
                        if st.session_state.page_state["info"]:
                            info_shown = True
                            st.session_state.page_state["info"].render(margin=True)
                            
                        ## Action Menu
                        action_menu = MuiActionMenu(
                            mode=["upload", "edit", "delete", "move"],
                            key="full_menu",
                            onClick=FileActionDialog.buildDialogOpener("main_dialog"),
                        ).render()
                        
                        no_upload = MuiActionMenu(
                            mode=["edit", "delete", "move"],
                            key="no_upload_menu",
                            onClick=FileActionDialog.buildDialogOpener("main_dialog"),
                        ).render()

                        ## BreadCrumb
                        def breadCrumbClickHandler(event: dict):
                            clicked_path = getEventValue(event)
                            logger.debug(f"breadCrumbClickHandler: {clicked_path}")

                            with st.session_state.lock:
                                clicked_path = getEventValue(event)
                                if clicked_path == curr_dir.path:
                                    action_menu.buildMenuOpener(
                                        file=st.session_state.page_state["curr_dir"]
                                    )(event)
                                else:
                                    clicked_path = replaceInitialSegment(clicked_path, "./")
                                    if st.session_state.page_state["menu"] == "user":
                                        args = deepcopy(user_default)
                                        args["path"] = clicked_path
                                    else:
                                        args = deepcopy(shared_default)
                                        args["path"] = clicked_path
                                    st.session_state.page_state["curr_dir"] = QuetzalFile(**args)
                                    st.session_state.page_state["info"] = None

                        MuiFilePathBreadcrumbs(
                            file=curr_dir,
                            key="curr_dir_breadcrumb",
                            onClick=breadCrumbClickHandler,
                        ).render()

                        ## File List
                        def fileListMoreHandler(event: dict):
                            file = getEventValue(event)
                            if file != None:
                                logger.debug("Calling MenuOpener")
                                no_upload.buildMenuOpener(file=file)(event)
                        
                        def fileClickHandler(event: dict):
                            with st.session_state.lock:
                                file = getEventValue(event)
                                if (
                                    st.session_state.page_state["info_file"] 
                                    and st.session_state.page_state["info_file"].path == file.path
                                ):
                                    st.session_state.page_state["info_file"] = None
                                else:
                                    st.session_state.page_state["info_file"] = file
                            
                        def fileDoubleClickHandler(event: dict):
                            with st.session_state.lock:
                                file = getEventValue(event)
                                if file.type == FileType.DIRECTORY:
                                    st.session_state.page_state["curr_dir"] = file
                                    st.session_state.page_state["info"] = None
                                else: # filtype = FILE
                                    st.session_state.page_state["info_file"] = file
                                
                            
                        filter_content = st.session_state.page_state["menu"]=="shared"
                        file_list = MuiFileList(
                            file_list=curr_dir.listFiles(
                                sharedOnly=filter_content,
                                excludeUser=filter_content
                            ),
                            max_height=f"calc(100% - 8rem)" if info_shown else f"calc(100% - 48px)", 
                            key="main",
                            onClickMore=fileListMoreHandler,
                            onClick=fileClickHandler,
                            onDoubleClick=fileDoubleClickHandler
                        ).render()
                        # file_list.render()

                        ## Click Focus Handler
                        on_focus_handler.setScanner(key="file_list")
                        on_focus_handler.registerHandler(keys="file_list", handler=lambda: MuiActionMenu.resetAnchor(
                                    exculde_keys=["full_menu", "no_upload_menu"]
                                ))
                        # on_focus_handler.registerHandler(file_list.onFocusOut)


    with info_c:
        with stylable_container(
            key="file_info",
            css_styles=f"""{{
                display: block;
                div {{
                    width: 100%;
                    height: auto;
                }}
                iframe {{
                    width: {INFO_WIDTH}px !important;
                    # height: 100px;
                }}
                
                .stSlider {{
                    position: absolute;
                    right: 0px !important;
                    width: {INFO_WIDTH}px !important;
                }}
                
                video {{
                    # position: absolute;
                    right: 0px !important;
                    width: {INFO_WIDTH}px !important;
                }}
            }}
            """,
        ):
            
            with stylable_container(
                key="file_info2",
                css_styles=f"""{{
                    display: block;
                    border-radius: 1rem 1rem 1rem 1rem;
                    height: calc({window_height}px - {padding} - 0.5rem) !important;
                    max-height: calc({window_height}px - {padding} - 0.5rem) !important;
                    position: absolute;
                    right: 0px !important;
                    width: {INFO_WIDTH}px !important;
                    background-color: white;
                }}
                """,
            ):
                video_placeholder_height = int(INFO_WIDTH / 16 * 9 + 0.5)
                if info_file and info_file.type == FileType.FILE:
                    infoHeight = f"calc({window_height}px - {padding} - {video_placeholder_height}px - 13rem) !important;"
                else:
                    infoHeight = f"calc({window_height}px - {padding} - {video_placeholder_height}px - 9.5rem) !important;"
                    
                def onVideoSelect(event):
                    with st.session_state.lock:
                        video_type = getEventValue(event)
                        if st.session_state.page_state["compare"]["project"] != curr_dir.getName():
                            st.session_state.page_state["compare"] = {"project": None, "database": None, "query": None}
                        
                        st.session_state.page_state["compare"]["project"] = curr_dir.getName()
                        st.session_state.page_state["compare"][video_type] = info_file.getName()
                
                def closeDetail(event):
                    file_list.onFocusOut()
                    st.session_state.page_state["info_file"] = None
                            
                MuiFileDetails(
                    file=info_file,
                    width=INFO_WIDTH,
                    infoHeight=infoHeight,
                    video_placeholder_height=video_placeholder_height,
                    key="main_dialog",
                    onClick=onVideoSelect,
                    onClose=closeDetail
                ).render()        


    

    FileActionDialog("main_dialog").render()
    
    print(st.session_state.DialogState)
# FileActionDialog("sub_dialog").render()

