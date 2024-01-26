import streamlit as st
import base64

# from streamlit_extras.stylable_container import stylable_container
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

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

logger.info("Streamlit Rerunning")

BORDER_RADIUS = "0.8rem"
BACKGROUND_COLOR = "#f8fafd"
PRIMARY_COLOR = "#c9e6fd"
EXAMPLE_LINK = "home/project_name/sub_project/name"
TEST_FILE_LIST = [
    {
        "name": "Project Name 1",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 1",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 2",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Project Name 2",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 3",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 4",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 3",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Video Name 5",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 4",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 6",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 5",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 7",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Project Name 6",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 8",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 7",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 9",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Project Name 8",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 10",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Project Name 9",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Very Long Video Name 11",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Very Very Long Project Name 10",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Video Name 12",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 11",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
]

def convert_to_quetzal_files(file_info):
    name = file_info["name"]
    type = file_info["type"]
    state = file_info["state"]
    visibility = state["visibility"]
    analysis_progress = state["analyzed"]
    permission = state["permission"]
    
    # Assuming the owner is the same for all files, using a placeholder
    owner = "jinho yi"
    
    quetzal_file = QuetzalFile(
        path=name,  # Using 'name' as 'path' for simplicity
        owner=owner,
        type=type,
        visibility=visibility,
        analysis_progress=analysis_progress,
        permission=permission
    )
    
    return quetzal_file

top_padding = "1rem"

st.set_page_config(layout="wide")
st.markdown(
    f"""
        <style>
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

float_init()


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
        "--dataset-root", default="../data", help="Root directory of datasets"
    )
    parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=True)
    parser.add_argument("--cuda_device", help="Select cuda device", default=0, type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root

    available_gpus = torch.cuda.device_count()
    print(f"Avaliable GPU={available_gpus}")
    if args.cuda and available_gpus > 0:
        cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
        torch_device = torch.device("cuda:" + str(cuda_device))
    else:
        torch_device = torch.device("cpu")

    print(torch_device)

    return dataset_root, cuda_device, torch_device


# Initialize Variable
dataset_root, cuda_device, torch_device = parse_args()
project_name = "example_mil19"
query_video = "P0200020.MP4"
database_video = "P0190019.MP4"
grounding_sam = None

MENU_WIDTH = 250
INFO_WIDTH = 320

# 0.174
# 0.223

window_height = streamlit_js_eval(
    js_expressions="screen.height", want_output=True, key="WIND_H"
)
padding = "11.5rem"

options = ["option1"] * 10
ITEM_HEIGHT = 24

curr_file = QuetzalFile(
    path=EXAMPLE_LINK,
    type=FileType.DIRECTORY,
    visibility=Visibility.SHARED,
    analysis_progress=AnalysisProgress.NONE,
    permission=Permission.FULL_WRITE,
    owner="jinhoy",
)

if "dialogOpen" not in st.session_state:
    st.session_state.dialogOpen = False

if "show" not in st.session_state:
    st.session_state.show = False

if "page_state" not in st.session_state:
    st.session_state.page_state = {"curr_dir": curr_file}


def createHandleClickOpen(file_event):
    def handleClick():
        st.session_state.show = True
        print(file_event)

    # st.session_state.dialogOpen = True
    # print(file_event)
    # st.session_state.show = True
    return handleClick


def handleClickOpen():
    # st.session_state.dialogOpen = True
    print("Im Called Here")
    st.session_state.show = True
    st.session_state.MuiDialogState["main"]["open"] = True


def handleClose(event, action):
    print("Dialog closed with action:", event, action)
    # st.session_state.dialogOpen = False


def click_away(event):
    print("Main Page: Clicked Menu Area\n")
    # print(st.session_state.MuiDialogState["main"]["open"])


on_focus_handler = MuiOnFocusHandler()

menu_c, files_c, info_c = st.columns([2, 100, 2])
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
                    onClick=FileActionDialog.buildDialogOpener(),
                ).render()
                # upload_menu.render()

                ## Upload Button
                UploadButton(
                    key="upload_button", onClick=upload_menu.buildMenuOpener(curr_file)
                ).render()

                ## Side bar Menu
                toggle_buttons = [
                    MuiToggleButton("user", "Home", "My Projects"),
                    MuiToggleButton("shared", "FolderShared", "Shared by Others"),
                ]
                MuiSideBarMenu(toggle_buttons=toggle_buttons, key="main_menu").render()

                on_focus_handler.setScanner(key="menu")
                on_focus_handler.registerHandler(keys=["menu"], handler=lambda: MuiActionMenu.resetAnchor("upload_action_menu"))
                # ## Click Focus Handler
                # focus_handler = MuiFocusHandler(element_key).render()
                # # focus_handler.render()
                # focus_handler.register(
                #     lambda x: MuiActionMenu.resetAnchor("upload_action_menu")
                # )


# with files_c:
#     with stylable_container(
#         key="slider",
#         css_styles=f"""{{
#                     display: block;
#                     div {{
#                         width: 100%;
#                         height: auto;
#                     }}
#                     iframe {{
#                         position: absolute;
#                         right: calc({INFO_WIDTH}px - 2%) !important;
#                         width: calc(104% - {INFO_WIDTH}px - {MENU_WIDTH}px) !important;
#                         height: calc({window_height}px - {padding});
#                     }}
#                     # position: absolute;
#                     # right: calc({INFO_WIDTH}px - 2%) !important;
#                     # width: calc(104% - {INFO_WIDTH}px - {MENU_WIDTH}px) !important;  
#                     # .stSlider {{
#                     #     position: absolute;
#                     #     right: calc({INFO_WIDTH}px - 2%) !important;
#                     #     width: calc(104% - {INFO_WIDTH}px - {MENU_WIDTH}px) !important;                   
#                     # }}
#                 }}
#                 """,
#     ):
#         with elements("files"):
#             with mui.Paper(
#                 variant="outlined",
#                 sx={
#                     "borderRadius": "0px",
#                     "border": "0px",
#                     "width": "100%",
#                     "height": f"calc({window_height}px - {padding})",
#                     "bgcolor": BACKGROUND_COLOR,
#                     "m": "0px",
#                     "p": "0px",
#                     "position": "absolute",
#                     "left": "0px",
#                     "top": "0px",
#                 },
#             ):
#                 with mui.Paper(
#                     variant="outlined",
#                     sx={
#                         "borderRadius": "1rem",
#                         "border": "0px",
#                         "height": f"calc({window_height}px - {padding} - {top_padding} - 0.5rem);",
#                         "bgcolor": "white",
#                         "padding": "0.5rem",
#                         "margin-right": "1rem",
#                         # "padding-right" : "2rem",
#                     },
#                 ):
#                     ## Action Menu
#                     action_menu = MuiActionMenu(
#                         mode=["upload", "edit", "delete"],
#                         key="full_menu",
#                         onClick=FileActionDialog.buildDialogOpener(),
#                     )
#                     action_menu.render()
                    
#                     no_upload = MuiActionMenu(
#                         mode=["edit", "delete"],
#                         key="no_upload_menu",
#                         onClick=FileActionDialog.buildDialogOpener(),
#                     )
#                     no_upload.render()

#                     ## BreadCrumb
#                     def breadCrumbClickHandler(event: dict):
#                         clicked_path = event["target"]["value"]
#                         if clicked_path == curr_file.path:
#                             action_menu.buildMenuOpener(file=curr_file)(event)
#                         else:
#                             logger.info(clicked_path)

#                     MuiFilePathBreadcrumbs(
#                         file=curr_file,
#                         key="curr_dir_breadcrumb",
#                         onClick=breadCrumbClickHandler,
#                     ).render()

#                     ## File List
#                     def fileListMoreHandler(event: dict):
#                         file = event["target"]["value"]
#                         if file != None:
#                             logger.debug("Calling MenuOpener")
#                             no_upload.buildMenuOpener(file=file)(event)
                    
#                     file_list = MuiFileList(
#                         file_list=map(convert_to_quetzal_files, TEST_FILE_LIST), 
#                         max_height="calc(100% - 48px)", 
#                         key="main",
#                         onClickMore=fileListMoreHandler,
#                     )
#                     file_list.render()

#                     ## Click Focus Handler
#                     on_focus_handler.setScanner(key="file_list")
#                     on_focus_handler.registerHandler(keys="file_list", handler=lambda: MuiActionMenu.resetAnchor(
#                                 exculde_keys=["full_menu", "no_upload_menu"]
#                             ))
#                     on_focus_handler.registerHandler(file_list.onFocusOut)

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
                        # "padding-right": f"{padding_right + inner_padding}rem",
                        "zIndex": 2,
                    },
                ):
                    ## Action Menu
                    action_menu = MuiActionMenu(
                        mode=["upload", "edit", "delete"],
                        key="full_menu",
                        onClick=FileActionDialog.buildDialogOpener(),
                    )
                    action_menu.render()
                    
                    no_upload = MuiActionMenu(
                        mode=["edit", "delete"],
                        key="no_upload_menu",
                        onClick=FileActionDialog.buildDialogOpener(),
                    )
                    no_upload.render()

                    ## BreadCrumb
                    def breadCrumbClickHandler(event: dict):
                        clicked_path = event["target"]["value"]
                        if clicked_path == curr_file.path:
                            action_menu.buildMenuOpener(file=curr_file)(event)
                        else:
                            logger.info(clicked_path)

                    MuiFilePathBreadcrumbs(
                        file=curr_file,
                        key="curr_dir_breadcrumb",
                        onClick=breadCrumbClickHandler,
                    ).render()

                    ## File List
                    def fileListMoreHandler(event: dict):
                        file = event["target"]["value"]
                        if file != None:
                            logger.debug("Calling MenuOpener")
                            no_upload.buildMenuOpener(file=file)(event)
                    
                    file_list = MuiFileList(
                        file_list=map(convert_to_quetzal_files, TEST_FILE_LIST), 
                        max_height="calc(100% - 48px)", 
                        key="main",
                        onClickMore=fileListMoreHandler,
                    )
                    file_list.render()

                    ## Click Focus Handler
                    on_focus_handler.setScanner(key="file_list")
                    on_focus_handler.registerHandler(keys="file_list", handler=lambda: MuiActionMenu.resetAnchor(
                                exculde_keys=["full_menu", "no_upload_menu"]
                            ))
                    on_focus_handler.registerHandler(file_list.onFocusOut)



with info_c:
    with stylable_container(
        key="file_info",
        css_styles=f"""{{
            display: block;
            # position: absolute;
            # right: 0px !important;
            # width: {INFO_WIDTH}px !important;
            div {{
                width: 100%;
                height: auto;
            }}
            iframe {{
                # position: absolute;
                # right: 0px !important;
                width: {INFO_WIDTH}px !important;
                height: 100px;
            }}
            
            .stSlider {{
                position: absolute;
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
                height: calc({window_height}px - {padding} - 0.5rem);
                position: absolute;
                right: 0px !important;
                width: {INFO_WIDTH}px !important;
                background-color: white;
            }}
            """,
        ):
            st.header("quetzal")
            st.slider(
                value=0,
                label="Info!",
                min_value=0,
                max_value=100,
                step=1,
                format=f"%d",
            )
            with elements("info_1"):
                with mui.Paper(
                    variant="outlined",
                    sx={
                        "borderRadius": 0,
                        "border": "0px",
                        "width": "100%",
                        "height": "5rem",
                        "bgcolor": BACKGROUND_COLOR,
                        "m": "0px",
                        "padding": "0",
                        # "position": "absolute",
                        "left": "0px",
                        "top": "0px",
                    },
                ):
                    with mui.Paper(
                        variant="outlined",
                        sx={
                            "borderRadius": "1rem",
                            "border": "0px",
                            "height": "5rem",
                            "bgcolor": "white",
                            "padding": "1rem",
                            # "margin-right": "1rem",
                            # "padding-right" : "2rem",
                        },
                    ):
                    
                        mui.Typography(
                            variant="h5",
                            component="div",
                            children=["Info"],
                            sx={"margin-bottom": "0.5rem"},
                        )
                
            st.title("WHAT")
            st.selectbox("hello", options=["what", "is","this"])

# with st.container(border=True):
#     if st.session_state.controller == "playback":
#         if st.session_state.warp:
#             query_img = overlay_query_frame_list[matches[0]]
#             db_img = db_frame_list[matches[1]]
#         else:
#             query_img = query_frame_list[matches[0]]
#             db_img = db_frame_list[matches[1]]
#     elif st.session_state.controller == "object":
#         if st.session_state.annotated_frame["idx"] == st.session_state.slider:
#             query_img = st.session_state.annotated_frame["query"]
#             db_img = st.session_state.annotated_frame["db"]
#         else:
#             query_img = query_frame_list[matches[0]]
#             db_img = db_frame_list[matches[1]]
#     # elif st.session_state.controller == "overlay":
#     #     query_img = overlay_query_frame_list[matches[0]]
#     #     db_img = db_frame_list[matches[1]]

#     query_img_base64 = f"data:image/jpeg;base64,{get_base64(query_img)}"
#     db_img_base64 = f"data:image/jpeg;base64,{get_base64(db_img)}"

#     imgc1, imgc2 = st.columns([1, 1], gap="small")
#     with imgc1:
#         display_frame(
#             label="Query Frame: " + query_video,
#             images=[query_img_base64],
#             frame_len=len(query_frame_list),
#             idx=matches[0],
#             fps=2,
#         )

#     with imgc2:
#         if st.session_state.controller != "overlay":
#             display_frame(
#                 label="Aligned Database Frame: " + database_video,
#                 images=[query_img_base64, db_img_base64],
#                 frame_len=len(db_frame_list),
#                 idx=matches[1],
#                 fps=6,
#             )

#     if BENCH_MARK:
#         st.info(f"⏱️ Image loaded in {time.time() - start_time:.4f} seconds")

#     controller_tab_ui()

# if st.session_state.playback:
#     st.session_state.next_frame = False
#     curr_wakeup_time = st.session_state.wakeup_time

#     sleep_duration = max(
#         0, (curr_wakeup_time - datetime.datetime.now()).total_seconds()
#     )
#     time.sleep(sleep_duration)

#     if (
#         st.session_state.wakeup_time == curr_wakeup_time
#     ):  ## no other instance modified it
#         st.session_state.wakeup_time += datetime.timedelta(seconds=0.5)
#         st.session_state.next_frame = True
#         st.rerun()
FileActionDialog().render()

# MuiActionMenu.resetAnchor()
