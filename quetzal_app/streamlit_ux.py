import streamlit as st
import base64
from streamlit_extras.stylable_container import stylable_container
from streamlit_image_comparison import image_comparison

from streamlit_extras.st_keyup import st_keyup
from streamlit_elements import elements, mui, html
import time
import datetime

# import streamlit_shadcn_ui as ui
import streamlit_antd_components as sac

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
               .block-container {{
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }}
                .stSlider {{
                    padding-left: 1rem;
                    padding-right: 1rem;
                }}
                [class^="st-emotion-cache-"] {{
                    gap: 0rem;
                    border-radius: {BORDER_RADIUS};
                }}
                # iframe {{
                #     position: relative;
                #     width: 100%;
                # }}
      </style>
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
    result = _run_alignment(dataset_root, route, query, db, False)
    return result

# Initialize Variable
dataset_root, cuda_device, torch_device = parse_args()
project_name = "mil19_orig"
query_video = "P0200020.MP4"
database_video = "P0190019.MP4"
grounding_sam = None

if "page_state" not in st.session_state:
    st.session_state.page_state = {
        "playback": {
            "slider": DEFAULT_SLIDER_VAL
        }, 
        "object": {
            "slider_box_th": DEFAULT_BOX_TH, 
            "slider_text_th": DEFAULT_TEXT_TH
        }
    }

if "matches" not in st.session_state:
    # st.session_state.matches = run_alignment(project_name, query_video, database_video)
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

# if "slider_box_th" not in st.session_state:
#     st.session_state.slider_box_th = DEFAULT_BOX_TH

# if "slider_text_th" not in st.session_state:
#     st.session_state.slider_text_th = DEFAULT_TEXT_TH

if "next_frame" not in st.session_state:
    st.session_state.next_frame = False

if "playback" not in st.session_state:
    st.session_state.playback = False

if "slider" not in st.session_state:
    st.session_state.slider = st.session_state.page_state["playback"]["slider"]
    st.session_state.slider_val = st.session_state.slider

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
SLIDE_MAX = len(query_frame_list)

FRAME_IDX = "Frame Index: {}/{}"
PLAYBACK_TIME = "Playback Time: {}/{}"
TEXT_ALIGN_LEFT = '<div style="text-align: left;"> {} </div>'
TEXT_ALIGN_RIGHT = '<div style="text-align: right;"> {} </div>'
FRAME_IDX_ST = TEXT_ALIGN_LEFT.format("&nbsp" + FRAME_IDX)
PLAYBACK_TIME_ST = TEXT_ALIGN_RIGHT.format(PLAYBACK_TIME + "&nbsp")
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
        st.session_state.text_prompt, 
        query_img_orig,
        QUERY_ANNOTATE_IMG,
        st.session_state.slider_box_th,
        st.session_state.slider_text_th
        )
    
    _run_detection(
        st.session_state.text_prompt, 
        database_img_aligned,
        DB_ANNOTATE_IMG,
        st.session_state.slider_box_th,
        st.session_state.slider_text_th
        )
    
    st.session_state.annotated_frame = {
        "query": QUERY_ANNOTATE_IMG,
        "db": DB_ANNOTATE_IMG,
        "idx": st.session_state.page_state["playback"]["slider"]
        }

def blend_img(idx, query, db): 
    return (query * idx + db * (1-idx)).astype(np.uint8)

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
        st.session_state.page_state[page]['slider'] = st.session_state.slider
    elif page == "object":
        st.session_state.page_state[page]['slider_box_th'] = st.session_state.slider_box_th
        st.session_state.page_state[page]['slider_text_th'] = st.session_state.slider_text_th

def loadPageState(page):
    if page == "playback":
        st.session_state.slider = st.session_state.page_state[page]['slider']
    elif page == "object":
        st.session_state.slider_box_th = st.session_state.page_state[page]['slider_box_th']
        st.session_state.slider_text_th = st.session_state.page_state[page]['slider_text_th']
        if "grounding_sam" not in st.session_state:
            st.session_state.grounding_sam = getGroundingSAMEngine(torch_device)
    elif page == "overlay":
        if "overlay_loaded" not in st.session_state:
            st.session_state.matches = _run_alignment(dataset_root, project_name, query_video, database_video, True)
            st.session_state.overlay_loaded = True
        load_overlay()

def handleController(event, controller):
    if controller != None:
        # print(st.session_state.page_state)
        savePageState(st.session_state.controller)
        # print(st.session_state.page_state)
        loadPageState(controller)
        # print(st.session_state.page_state)
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

# def change_slider(val=0, key="slider"):
#     if key == "slider":
#         set_slider(st.session_state.slider + val)
#     else:
#         st.session_

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

def playback_ui():
    cc1, cc2, cc3 = st.columns([8, 16, 1])
    with cc2:
        st.slider(
            value=st.session_state.page_state['playback']['slider'],
            label="Query Frame Index",
            min_value=SLIDE_MIN,
            max_value=SLIDE_MAX,
            step=1,
            format=f"%d",
            key="slider",
            on_change=change_slider,
        )
    with cc3:
        st.number_input(
            value=st.session_state.page_state['playback']['slider'],
            label=" ",
            key="slider_val",
            step=1,
            min_value=SLIDE_MIN,
            max_value=SLIDE_MAX,
            on_change=lambda: set_slider(st.session_state.slider_val),
        )
    with cc1:
        with elements("element1"):
            with mui.Stack(
                spacing=1,
                direction="row",
                sx={"my": 0, "maxHeight": 48, "minWidth": 244},
                alignItems="center",
                justifyContent="center",
                flexWrap="wrap",
            ):
                # with mui.Box(sx={"display": "flex", "flexDirection": "row", "alignItems": "center","justifyContent": "center", }):
                with mui.IconButton(onClick=lambda: change_slider(-5), sx={"fontSize": 48}):
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

def object_detection_ui():
    with elements("object_detection_controller"):
        with mui.Stack(
            spacing=2,
            direction="row",
            sx={"my": 0, "maxHeight": 48, "minWidth": 244},
            alignItems="center",
            justifyContent="center",
            flexWrap="wrap",
        ):
            mui.TextField(
                id="outlined-search", 
                # label="Search field", 
                type="search", 
                label="Detection Prompt [Seperate unique classes with '.', e.g: cat . dog . chair]",
                color="secondary",
                onChange=lazy(updateClass),
                defaultValue=st.session_state.text_prompt,
                sx={"flexGrow": 1, "borderRadius": "0.5rem"},
                )
            
            mui.Button(
                children="Detect",
                variant="contained", 
                startIcon=mui.icon.Search(),
                onClick=run_detection,
                size="large",
                sx={"bgcolor":"grey.800", "borderRadius": "0.5rem"}
                )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.slider(
            value=st.session_state.page_state['object']['slider_box_th'],
            label="Box Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format=f"%0.2f",
            key="slider_box_th",
        )
    with cc2:
        st.slider(
            value=st.session_state.page_state['object']['slider_text_th'],
            label="Text Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format=f"%0.2f",
            key="slider_text_th",
        )
    # keywords = st_tags(
    #     label='Enter Detection Classes',
    #     text='Press enter to add more',
    #     value=['Objects'],
    #     maxtags = -1,
    #     key='1')
    # st.write(keywords)

def display_frame_st(label, image, frame_len, idx, fps):
    total_time, show_hours = format_time(
        frame_len / fps, show_hours=False, final_time=True
    )
    curr_time, _ = format_time(idx / fps, show_hours)

    with st.container(border=False):
        st.markdown(TEXT_ALIGN_LEFT.format(f"&nbsp{label}"), unsafe_allow_html=True)
        st.image(image, use_column_width="always")

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown(FRAME_IDX_ST.format(idx, frame_len), unsafe_allow_html=True)
        with cc2:
            st.markdown(
                PLAYBACK_TIME_ST.format(curr_time, total_time), unsafe_allow_html=True
            )
        st.text("")

def display_frame_mui(label, image, frame_len, idx, fps):
    card_style = {
        "borderRadius": BORDER_RADIUS,
        # "backgroundColor": "grey.800",  # Dark background
        # "color": "white",  # Light text
    }

    total_time, show_hours = format_time(
        frame_len / fps, show_hours=False, final_time=True
    )
    curr_time, _ = format_time(idx / fps, show_hours)

    with elements("card_" + label):
        mui.Typography(
            label,
            sx={"padding-bottom": 4, "textAlign": "center", "fontFamily": "sans-serif"},
        )

        with mui.Card(variant="outlined", sx=card_style):
            mui.CardMedia(
                component="img",
                src=f"data:image/jpeg;base64,{get_base64(image)}",
            )
            with mui.CardContent():
                with mui.Stack(
                    spacing=1,
                    direction="row",
                    alignItems="center",
                    justifyContent="space-between",
                    flexWrap="wrap",
                ):
                    mui.Typography(
                        FRAME_IDX.format(idx, frame_len),
                        sx={"height": 12, "fontFamily": "sans-serif"},
                    )
                    mui.Typography(
                        PLAYBACK_TIME.format(curr_time, total_time),
                        sx={
                            "height": 12,
                            "textAlign": "right",
                            "fontFamily": "sans-serif",
                        },
                    )

st.title("Quetzal")
st.markdown("&nbsp;")

with st.container(border=True):
    if st.session_state.controller == "playback":
        query_img = query_frame_list[matches[0]]
        db_img = db_frame_list[matches[1]]
    elif st.session_state.controller == "object":
        if st.session_state.annotated_frame["idx"] == st.session_state.slider:
            query_img = st.session_state.annotated_frame["query"]
            db_img = st.session_state.annotated_frame["db"]
        else:
            query_img = query_frame_list[matches[0]]
            db_img = db_frame_list[matches[1]]
    elif st.session_state.controller == "overlay":
        query_img = overlay_query_frame_list[matches[0]]
        db_img = db_frame_list[matches[1]]

    
    imgc1, imgc2 = st.columns([1, 1], gap="small")
    with imgc1:
        display_frame_mui(
            label="Query Frame: " + query_video,
            image=query_img,
            frame_len=len(query_frame_list),
            idx=matches[0],
            fps=2
        )
       
    with imgc2:
        if st.session_state.controller != "overlay":
            display_frame_mui(
                label="Aligned Database Frame: " + database_video,
                image=db_img,
                frame_len=len(db_frame_list),
                idx=matches[1],
                fps=6
            )
        if st.session_state.controller == "overlay":
            # st.markdown(TEXT_ALIGN_LEFT.format(f"&nbsp Image Comparison"), unsafe_allow_html=True)
            st.subheader("Image Comparison")
            image_comparison(
                img1=query_img,
                img2=db_img,
                label1="Query Frame: " + query_video,
                label2="Aligned Databse Frame: " + database_video,
                starting_position=50,
                show_labels=True,
                width=780
            )

    # imgc1, imgc2 = st.columns([1,1], gap="small")
    # with imgc1: display_frame_st("Query Frame", query_frame_list, matches[0], 2)
    # with imgc2: display_frame_st("Aligned Database Frame", db_frame_list, matches[1], 6)


    if BENCH_MARK:
        st.info(f"⏱️ Image loaded in {time.time() - start_time:.4f} seconds")

    # controller_tab = sac.tabs(
    #     [
    #         sac.TabsItem(label='apple'),
    #         sac.TabsItem(icon='google'),
    #         sac.TabsItem(label='github', icon='github')
    #     ],
    #     align='center',
    #     return_index=True
    # )

    with elements("tabs"):
        with mui.ToggleButtonGroup(
            value=st.session_state.controller,
            onChange=handleController,
            aria_label="text alignment",
            exclusive=True,
            sx={
                "gap": "1rem",
                "& .MuiToggleButtonGroup-grouped": {
                    # "mx": "0.5rem",
                    "border": 0,
                    "bgcolor": 'grey.200',
                    "gap": "0.5rem",
                    "py": "0.2rem",
                    "&:not(:last-of-type)": {
                         "borderRadius": "0.5rem",
                    },
                    "&:not(:first-of-type)": {
                        "borderRadius": "0.5rem",
                    }
                },
                "& .MuiToggleButton-root": {
                    "&.Mui-selected": {
                        "color": "white",
                        "bgcolor": "black",
                        "&:hover" : {
                            "bgcolor": "black"
                        }
                    },
                    "&:hover" : {
                        "bgcolor": "grey.300"
                    }

                }
                
            },
        ):
            with mui.ToggleButton(value="playback", aria_label="left aligned"):
                mui.icon.PlayArrow()
                mui.Typography("Playback Control", sx={"fontFamily": "sans-serif", "textTransform": 'capitalize'})

            with mui.ToggleButton(value="object", aria_label="centered"):
                mui.icon.CenterFocusStrong()
                mui.Typography("Object Detection", sx={"fontFamily": "sans-serif", "textTransform": 'capitalize'})

            with mui.ToggleButton(value="overlay", aria_label="right aligned"):
                mui.icon.Compare()
                mui.Typography("Overlay Comparison", sx={"fontFamily": "sans-serif", "textTransform": 'capitalize'})

    if "text_prompt" not in st.session_state:
        st.session_state.text_prompt = ""

    def updateClass(event):
        st.session_state.text_prompt = event["target"]["value"]

    with st.container(border=True):
        if st.session_state.controller == "playback" or st.session_state.controller == "overlay" :
            playback_ui()
        if st.session_state.controller == "object":
            object_detection_ui()
            
                  

                    
            #         with mui.IconButton(onClick=toggle_play, sx={"fontSize": 48}):
            #             mui.icon.Search(fontSize="inherit")
            # # cc1, cc2 = st.columns([8,3])
            # with cc1:
            #     st_keyup(label="Enter Class to Detect", key="class_text_input", on_change=updateClass, value=st.session_state.text_prompt)
            # with cc2:

                

    # with elements("element2"):
    #     event.Hotkey("space", toggle_play)
    #     event.Hotkey("right", lambda :change_slider(1))
    #     event.Hotkey("left", lambda :change_slider(-1))
        
    #     list = [
    #         mui.ListItem(
    #             disablePadding=True,
    #             children=mui.ListItemButton(
    #                 mui.ListItemIcon(mui.icon.Inbox), mui.ListItemText(primary="Inbox")
    #             ),
    #         )
    #     ]
    #     mui.List(children=list, dense=False)


        # with mui.Stack(spacing=2, direction="row", sx={"mb": 1, "minWidth": 24}, alignItems="center", justifyContent="center"):
        #     mui.IconButton(
        #         children=mui.icon.FastRewind(fontSize="inherit"),
        #         onClick=lambda x:change_slider(-5),
        #         sx={"fontSize": 60}
        #     )

        #     mui.IconButton(
        #         children=mui.icon.SkipPrevious(fontSize="inherit"),
        #         onClick=lambda x:change_slider(-1),
        #         sx={"fontSize": 60}
        #     )

        #     with mui.IconButton(onClick=toggle_play, sx={"fontSize": 60}):
        #         if st.session_state.playback:
        #             startIcon= mui.icon.PauseCircle(fontSize="inherit")
        #         else:
        #             startIcon= mui.icon.PlayCircle(fontSize="inherit")

        #     mui.IconButton(
        #         children=mui.icon.SkipNext(fontSize="inherit"),
        #         onClick=lambda x:change_slider(1),
        #         sx={"fontSize": 60}
        #     )

        #     mui.IconButton(
        #         children=mui.icon.FastForward(fontSize="inherit"),
        #         onClick=lambda x:change_slider(5),
        #         sx={"fontSize": 60}
        #     )

        # mui.Button(
        #     startIcon=mui.icon.FastRewind,
        #     children="-5",
        #     variant="outlined",
        #     # fullWidth=True,
        #     size="large",
        #     onClick=lambda x:change_slider(-5)
        # )

        # mui.Button(
        #     startIcon=mui.icon.SkipPrevious,
        #     children="-1",
        #     variant="contained",
        #     # fullWidth=True,
        #     size="large",
        #     onClick=lambda x:change_slider(-1)
        # )
        # mui.Button(
        #     startIcon= mui.icon.PauseCircle if st.session_state.playback else mui.icon.PlayCircle,
        #     # children="",
        #     variant="contained",
        #     # fullWidth=True,
        #     size="large",
        #     onClick=toggle_play,
        # )
        # mui.Button(
        #     startIcon=mui.icon.SkipNext(sx={"minHeight": "20%", "minWidth": 48}),
        #     children="+1",
        #     variant="contained",
        #     # fullWidth=True,
        #     size="large",
        #     onClick=lambda x:change_slider(1),
        #     # disableRipple=True,
        #     sx={
        #         "bgcolor": 'grey.400',
        #         "borderRadius": 2,
        #         "borderColor":  'grey.400',
        #         "boxShadow": 0,
        #         "p": 1,
        #     },
        # )
        # mui.Button(
        #     startIcon=mui.icon.FastForward,
        #     children="+5",
        #     variant="outlined",
        #     # fullWidth=True,
        #     size="large",
        #     onClick=lambda x:change_slider(5)
        # )


        # mui.Box(
        #     "Some text in a styled box",
        #     sx={
        #         "bgcolor": "background.paper",
        #         "boxShadow": 1,
        #         "borderRadius": 2,
        #         "p": 2,
        #         "minWidth": 300,
        #     }
        # )

        # with mui.Paper:
        #     with mui.Typography:
        #         html.p("Hello world")
        #         html.p("Goodbye world")

        # # mui.Slider(aria_label="Volume", value=0, onChange=slider_callback)

        # with mui.Paper(elevation=3, variant="outlined", square=True):
        #     mui.TextField(
        #         label="My text input",
        #         defaultValue="Type here",
        #         variant="outlined",
        #     )

        # # If you must pass a parameter which is also a Python keyword, you can append an
        # # underscore to avoid a syntax error.
        # #
        # # <Collapse in />

        # mui.Collapse(in_=True)

        # mui.Stack(spacing=2, direction="row", sx={"mb": 1}, alignItems="center", children=component)

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
