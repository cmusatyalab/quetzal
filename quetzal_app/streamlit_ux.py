import streamlit as st
import base64

# from streamlit_extras.stylable_container import stylable_container
from streamlit_image_comparison import image_comparison

# from streamlit_javascript import st_javascript

from streamlit_elements import elements, mui, html
from quetzal_app.elements.mui_components import *
import time
import datetime
from streamlit_extras.stylable_container import stylable_container

# import streamlit_shadcn_ui as ui
# import streamlit_antd_components as sac

from streamlit_tags import st_tags
from streamlit_elements import event
from streamlit_elements import lazy

from quetzal.video import *
from quetzal.align_frames import align_video_frames, align_frame_pairs
from quetzal_app.utils.utils import *
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


TEMP_MATCH_FILENAME = "./match_obj.pkl"
BENCH_MARK = False
BORDER_RADIUS = "0.8rem"
DEFAULT_BOX_TH = 0.3
DEFAULT_TEXT_TH = 0.25
DEFAULT_SLIDER_VAL = 0

if BENCH_MARK:
    start_time = time.time()

st.set_page_config(layout="wide")
st.markdown(
    f"""
        <style>
               .block-container {{ /* Removes streamlit default white spaces in the main window*/
                    padding-top: 1.5rem;
                    # padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
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


def _run_alignment(
    dataset_root, project_name, database_video_name, query_video_name, overlay
):
    ## Load DTW and VLAD Features ##
    database_video = DatabaseVideo(
        datasets_dir=dataset_root,
        project_name=project_name,
        video_name=database_video_name,
    )
    query_video = QueryVideo(
        datasets_dir=dataset_root,
        project_name=project_name,
        video_name=query_video_name,
    )

    db_frame_list = database_video.get_frames()
    query_frame_list = query_video.get_frames()
    overlay_query_frame_list = query_frame_list

    if not overlay:
        matches = align_video_frames(
            database_video=database_video,
            query_video=query_video,
            torch_device=torch_device,
        )
    else:
        matches, overlay_query_frame_list = align_frame_pairs(
            database_video=database_video,
            query_video=query_video,
            torch_device=torch_device,
        )

    return {
        "matches": matches,
        "query_frame_list": query_frame_list,
        "db_frame_list": db_frame_list,
        "overlay_query_frame_list": overlay_query_frame_list,
    }


def run_alignment(route, query, db):
    result = _run_alignment(dataset_root, route, db, query, True)
    return result


# Initialize Variable
dataset_root, cuda_device, torch_device = parse_args()
project_name = "example_mil19"
query_video = "P0200020.MP4"
database_video = "P0190019.MP4"
grounding_sam = None

if "page_state" not in st.session_state:
    st.session_state.page_state = {
        "playback": {"slider": DEFAULT_SLIDER_VAL},
        "object": {
            "slider_box_th": DEFAULT_BOX_TH,
            "slider_text_th": DEFAULT_TEXT_TH,
            "class_prompts": ["objects"],
        },
    }

if "matches" not in st.session_state:
    st.session_state.matches = run_alignment(project_name, query_video, database_video)

    with open(TEMP_MATCH_FILENAME, "wb") as f:
        pickle.dump(st.session_state.matches, f)

    with open(TEMP_MATCH_FILENAME, "rb") as file:
        st.session_state.matches = pickle.load(file)
    file.close()

if "query_video" not in st.session_state:
    st.session_state.query_video = QueryVideo(
        datasets_dir=dataset_root, project_name=project_name, video_name=query_video
    )

if "database_video" not in st.session_state:
    st.session_state.database_video = QueryVideo(
        datasets_dir=dataset_root, project_name=project_name, video_name=database_video
    )

if "controller" not in st.session_state:
    st.session_state.controller = "playback"
    
if "warp" not in st.session_state:
    st.session_state.warp = True

if "next_frame" not in st.session_state:
    st.session_state.next_frame = False

if "playback" not in st.session_state:
    st.session_state.playback = False

if "slider" not in st.session_state:
    st.session_state.slider = st.session_state.page_state["playback"]["slider"]
    st.session_state.slider_val = st.session_state.slider
    
if "anchorEl" not in st.session_state:
    st.session_state.anchorEl = None
    
if "class_prompts" not in st.session_state:
    st.session_state.class_prompts = ["objects"]
    
if "do_rerun" not in st.session_state:
    st.session_state.do_rerun = False

if st.session_state.do_rerun:
    st.session_state.do_rerun = False
    st.rerun()
    
# Load Variables
query_frame_list = st.session_state.matches.get("query_frame_list")
overlay_query_frame_list = st.session_state.matches.get("overlay_query_frame_list")
db_frame_list = st.session_state.matches.get("db_frame_list")
matches = st.session_state.matches.get("matches")[st.session_state.slider]

if "annotated_frame" not in st.session_state:
    st.session_state.annotated_frame = {
        "query": query_frame_list[matches[0]],
        "db": db_frame_list[matches[1]],
        "idx": -1,
    }

if BENCH_MARK:
    st.info(f"⏱️ Image loaded in {time.time() - start_time:.4f} seconds")

# GUI
SLIDE_MIN = 0
SLIDE_MAX = len(query_frame_list) - 1

FRAME_IDX = "Frame Index: {}/{}"
PLAYBACK_TIME = "Playback Time: {}/{}"
QUERY_ANNOTATE_IMG = "../tmp/annotated_query.jpg"
DB_ANNOTATE_IMG = "../tmp/annotated_db.jpg"
BLENDED_IMG = "../tmp/blended.jpg"


@st.cache_resource
def getGroundingSAMEngine(torch: torch.device):
    return GoundingSAMEngine(torch)


# @st.cache_data
def _run_detection(text_prompt, input_img, output_file, box_threshold, text_threshold):
    grounding_sam = st.session_state.grounding_sam
    grounding_sam.generate_masked_images(
        input_img, text_prompt, output_file, box_threshold, text_threshold
    )


def run_detection():
    query_img_orig = query_frame_list[matches[0]]
    database_img_aligned = db_frame_list[matches[1]]

    _run_detection(
        # st.session_state.text_prompt,
        " . ".join(st.session_state.class_prompts),
        query_img_orig,
        QUERY_ANNOTATE_IMG,
        st.session_state.slider_box_th,
        st.session_state.slider_text_th,
    )

    _run_detection(
        # st.session_state.text_prompt,
        " . ".join(st.session_state.class_prompts),
        database_img_aligned,
        DB_ANNOTATE_IMG,
        st.session_state.slider_box_th,
        st.session_state.slider_text_th,
    )

    st.session_state.annotated_frame = {
        "query": QUERY_ANNOTATE_IMG,
        "db": DB_ANNOTATE_IMG,
        "idx": st.session_state.page_state["playback"]["slider"],
    }


def blend_img(idx, query, db):
    return (query * idx + db * (1 - idx)).astype(np.uint8)


def load_overlay():
    overlay_query_frame_list = st.session_state.matches["overlay_query_frame_list"]
    query_idx_orig, database_index_aligned = matches
    query_img = cv2.imread(overlay_query_frame_list[query_idx_orig])
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    db_img = cv2.imread(db_frame_list[database_index_aligned])
    db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)

    query_idx_orig, database_index_aligned = matches
    query_img = cv2.imread(overlay_query_frame_list[query_idx_orig])
    db_img = cv2.imread(db_frame_list[database_index_aligned])
    blended = blend_img(0.5, query_img, db_img)
    cv2.imwrite(BLENDED_IMG, blended)


def savePageState(page):
    if page == "playback":
        st.session_state.page_state[page]["slider"] = st.session_state.slider
    elif page == "object":
        st.session_state.page_state[page][
            "slider_box_th"
        ] = st.session_state.slider_box_th
        st.session_state.page_state[page][
            "slider_text_th"
        ] = st.session_state.slider_text_th
        st.session_state.page_state[page][
            "class_prompts"
        ] = st.session_state.class_prompts


def loadPageState(page):
    if page == "playback":
        st.session_state.slider = st.session_state.page_state[page]["slider"]
    elif page == "object":
        st.session_state.slider_box_th = st.session_state.page_state[page][
            "slider_box_th"
        ]
        st.session_state.slider_text_th = st.session_state.page_state[page][
            "slider_text_th"
        ]
        st.session_state.class_prompts = st.session_state.page_state[page][
            "class_prompts"
        ]
        if "grounding_sam" not in st.session_state:
            st.session_state.grounding_sam = getGroundingSAMEngine(torch_device)

def handleController(event, controller):
    if controller != None:
        savePageState(st.session_state.controller)
        loadPageState(controller)
        st.session_state.controller = controller


def set_slider(val):
    orig = st.session_state.slider
    st.session_state.slider = val
    if st.session_state.slider < SLIDE_MIN:
        st.session_state.slider = SLIDE_MIN
    elif st.session_state.slider >= SLIDE_MAX:
        st.session_state.slider = SLIDE_MAX
        if st.session_state.playback:
            toggle_play()

    st.session_state.slider_val = st.session_state.slider


def change_slider(val=0):
    set_slider(st.session_state.slider + val)

def toggle_play():
    st.session_state.playback = not st.session_state.playback
    if st.session_state.playback:
        st.session_state.wakeup_time = datetime.datetime.now() + datetime.timedelta(
            seconds=0.5
        )


if st.session_state.playback and st.session_state.next_frame:
    change_slider(1)

if BENCH_MARK:
    start_time = time.time()


def controller_tab_ui():
    toggle_buttons_style = {
        "gap": "1rem",
        "& .MuiToggleButtonGroup-grouped": {
            "border": 0,
            "bgcolor": "grey.200",
            "gap": "0.5rem",
            "py": "0.2rem",
            "&:not(:last-of-type)": {
                "borderRadius": "0.5rem",
            },
            "&:not(:first-of-type)": {
                "borderRadius": "0.5rem",
            },
        },
        "& .MuiToggleButton-root": {
            "&.Mui-selected": {
                "color": "white",
                "bgcolor": "black",
                "&:hover": {"bgcolor": "black"},
            },
            "&:hover": {"bgcolor": "grey.300"},
        },
    }
    toggle_buttons = [
        MuiToggleButton("playback", "PlayArrow", "Playback Control"),
        MuiToggleButton("object", "CenterFocusStrong", "Object Detection"),
        # MuiToggleButton("overlay", "Compare", "Overlay Comparison"),
    ]
    
    def handleSwitch():
        st.session_state.warp = not st.session_state.warp
    
    with elements("tabs"):
        with mui.Stack(
            spacing=2,
            direction="row",
            alignItems="start",
            justifyContent="space-between",
            sx={"my": 0, "maxHeight": "calc(30.39px - 22.31px + 1rem)"},
        ):
            with mui.ToggleButtonGroup(
                value=st.session_state.controller,
                onChange=handleController,
                exclusive=True,
                sx=toggle_buttons_style,
            ):
                for button in toggle_buttons:
                    button.render()
                    
            with mui.Stack(
                direction="row", 
                spacing=0, 
                alignItems="start",
                sx={"maxHeight": "calc(30.39px - 22.31px + 1rem)"}
            ):
                # mui.icon.Compare(sx={"py": "7px"})
                mui.Typography("Image Warp", sx={"ml": "0.3rem", "py": "7px"})
                mui.Switch(
                    checked=st.session_state.warp,
                    onChange=handleSwitch
                )

    with st.container(border=True):
        if (
            st.session_state.controller == "playback"
            # or st.session_state.controller == "overlay"
        ):
            playback_ui()
        if st.session_state.controller == "object":
            object_detection_ui()


def playback_ui():
    cc1, cc2, cc3 = st.columns([25, 50, 1])
    with cc1:
        PADDING = 8
        GAP = 8
        ICON_SIZE = 48
        BUTTON_SIZE = ICON_SIZE + 2 * PADDING
        PLAYER_CONTROLER_W = BUTTON_SIZE * 5 + GAP * (5 - 1) + 2 * PADDING
        with stylable_container(
            key="playback_controller",
            css_styles=f"""{{
                    display: block;
                    div {{
                        min-width: {PLAYER_CONTROLER_W}px;
                    }}
                    iframe {{
                        min-width: {PLAYER_CONTROLER_W}px;
                    }}
                }}
                """,
        ):
            with elements("playback_controller_element"):
                with mui.Stack(
                    spacing=1,
                    direction="row",
                    sx={
                        "my": 0,
                        "maxHeight": f"calc({BUTTON_SIZE}px - 22.31px + {PADDING}px)",
                        "minWidth": BUTTON_SIZE * 5 + GAP * (5 - 1),
                    },
                    alignItems="start",
                    justifyContent="center",
                ):
                    with mui.IconButton(
                        onClick=lambda: change_slider(-5), sx={"fontSize": 48}
                    ):
                        mui.icon.KeyboardDoubleArrowLeft(fontSize="inherit")

                    mui.IconButton(
                        children=mui.icon.KeyboardArrowLeft(fontSize="inherit"),
                        onClick=lambda: change_slider(-1),
                        sx={"fontSize": 48},
                    )

                    with mui.IconButton(onClick=toggle_play, sx={"fontSize": 48}):
                        if st.session_state.playback:
                            mui.icon.Pause(fontSize="inherit")
                        else:
                            mui.icon.PlayArrow(fontSize="inherit")

                    mui.IconButton(
                        children=mui.icon.KeyboardArrowRight(fontSize="inherit"),
                        onClick=lambda: change_slider(1),
                        sx={"fontSize": 48},
                    )

                    mui.IconButton(
                        children=mui.icon.KeyboardDoubleArrowRight(fontSize="inherit"),
                        onClick=lambda: change_slider(5),
                        sx={"fontSize": 48},
                    )
    with cc2:
        with stylable_container(
            key="slider",
            css_styles=f"""{{
                    display: block;
                    .stSlider {{
                        position: absolute;
                        right: calc(45px - 2%) !important;
                        width: calc(102% - 45px) !important;
                        max-width: calc(152% - 45px - {PLAYER_CONTROLER_W}px) !important;   
                        padding-right: 0.5rem;
                    }}
                    
                }}
                """,
        ):
            st.slider(
                value=st.session_state.page_state["playback"]["slider"],
                label="Query Frame Index",
                min_value=SLIDE_MIN,
                max_value=SLIDE_MAX,
                step=1,
                format=f"%d",
                key="slider",
                on_change=change_slider,
            )
    with cc3:
        with stylable_container(
            key="slider_value",
            css_styles=f"""{{
                    display: block;
                    .stNumberInput {{
                        position: absolute;
                        right: 0px !important;
                        width: 45px !important;
                    }}
                }}
                """,
        ):
            st.number_input(
                value=st.session_state.page_state["playback"]["slider"],
                label=" ",
                key="slider_val",
                step=1,
                min_value=SLIDE_MIN,
                max_value=SLIDE_MAX,
                on_change=lambda: set_slider(st.session_state.slider_val),
            )


def object_detection_ui():
    c1, c2 = st.columns([100, 1])

    with c1:
        with stylable_container(
            key="class_prompt_list",
            css_styles="""{
                display: block;
                padding: 0.5em 0.5em 0em; /*top right&left bottom*/
                width: calc(101% - 133px);
                }
            """,
        ):
            with st.container():
                st_tags(
                    label=" ",
                    text="Detection Prompt: Press enter to add more class",
                    value=st.session_state.class_prompts,
                    suggestions=[],
                    maxtags=10,
                    key="class_prompts",
                )

    with c2:
        with stylable_container(
            key="object_detect_button",
            css_styles="""{
                    display: block;
                    position: absolute;
                    width: 133px;
                    right: 0px;
                    div {
                        width: 133px;
                        height: auto;
                    }
                    iframe {
                        width: 133px;
                        height: 57px;
                    }
                }
                """,
        ):
            with elements("object_detection_controller"):
                mui.Button(
                    children="Detect",
                    variant="contained",
                    startIcon=mui.icon.Search(),
                    onClick=run_detection,
                    size="large",
                    sx={
                        "bgcolor": "grey.800",
                        "borderRadius": "0.5rem",
                        "width": "117.14px",
                    },
                )


    cc1, cc2 = st.columns(2)
    with cc1:
        st.slider(
            value=st.session_state.page_state["object"]["slider_box_th"],
            label="Box Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format=f"%0.2f",
            key="slider_box_th",
        )
    with cc2:
        st.slider(
            value=st.session_state.page_state["object"]["slider_text_th"],
            label="Text Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format=f"%0.2f",
            key="slider_text_th",
        )


def display_frame(label, images, frame_len, idx, fps):
    total_time, show_hours = format_time(
        frame_len / fps, show_hours=False, final_time=True
    )
    curr_time, _ = format_time(idx / fps, show_hours)

    image_frame(
        image_urls=images,
        captions=[
            FRAME_IDX.format(idx, frame_len),
            PLAYBACK_TIME.format(curr_time, total_time),
        ],
        label=label,
        starting_point=0,
        dark_mode=False,
        key="image_comparison" + str(fps),
    )

def handle_popover_open(event):
    st.session_state.anchorEl = True

def handle_popover_close():
    st.session_state.anchorEl = None

title, tooltip = st.columns([1,1])
with title:
    st.title("Quetzal")
with tooltip:
    PADDING = 62
    with elements("tooltip"):
        with mui.Stack(
                direction="row", 
                spacing=0, 
                alignContent="end",
                justifyContent="center",
                sx={"maxHeight": f"{PADDING}px"}
            ):
            mui.icon.Compare(
                sx={"pt": f"{PADDING-8}px", "px": "1rem"}, 
                onMouseEnter=handle_popover_open,
                onMouseLeave=handle_popover_close
            )
        with mui.Popover(
            id="mouse-over-popover",
            sx={"pointerEvents": "none"},
            open=st.session_state.anchorEl,
            anchorPosition={ "top": 0, "left": 0 },
            anchorOrigin={"vertical": "bottom", "horizontal": "center"},
            transformOrigin={"vertical": "bottom", "horizontal": "center"},
            onClose=handle_popover_close,
            disableRestoreFocus=True,
        ):
            mui.Typography('Use the slider on the "Aligned Data Frame" to compare the matched images', sx={"p": 1})


with st.container(border=True):
    if st.session_state.controller == "playback":
        if st.session_state.warp:
            query_img = overlay_query_frame_list[matches[0]]
            db_img = db_frame_list[matches[1]]
        else:
            query_img = query_frame_list[matches[0]]
            db_img = db_frame_list[matches[1]]
    elif st.session_state.controller == "object":
        if st.session_state.annotated_frame["idx"] == st.session_state.slider:
            query_img = st.session_state.annotated_frame["query"]
            db_img = st.session_state.annotated_frame["db"]
        else:
            query_img = query_frame_list[matches[0]]
            db_img = db_frame_list[matches[1]]
    # elif st.session_state.controller == "overlay":
    #     query_img = overlay_query_frame_list[matches[0]]
    #     db_img = db_frame_list[matches[1]]

    query_img_base64 = f"data:image/jpeg;base64,{get_base64(query_img)}"
    db_img_base64 = f"data:image/jpeg;base64,{get_base64(db_img)}"

    imgc1, imgc2 = st.columns([1, 1], gap="small")
    with imgc1:
        display_frame(
            label="Query Frame: " + query_video,
            images=[query_img_base64],
            frame_len=len(query_frame_list),
            idx=matches[0],
            fps=2,
        )

    with imgc2:
        if st.session_state.controller != "overlay":
            display_frame(
                label="Aligned Database Frame: " + database_video,
                images=[query_img_base64, db_img_base64],
                frame_len=len(db_frame_list),
                idx=matches[1],
                fps=6,
            )

    if BENCH_MARK:
        st.info(f"⏱️ Image loaded in {time.time() - start_time:.4f} seconds")

    controller_tab_ui()

if st.session_state.playback:
    st.session_state.next_frame = False
    curr_wakeup_time = st.session_state.wakeup_time

    sleep_duration = max(
        0, (curr_wakeup_time - datetime.datetime.now()).total_seconds()
    )
    time.sleep(sleep_duration)

    if (
        st.session_state.wakeup_time == curr_wakeup_time
    ):  ## no other instance modified it
        st.session_state.wakeup_time += datetime.timedelta(seconds=0.5)
        st.session_state.next_frame = True
        st.rerun()
