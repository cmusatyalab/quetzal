import streamlit as st
import os
from streamlit import session_state as ss

from streamlit_elements import elements, mui
from streamlit_label_kit import detection
from glob import glob
from quetzal_app.elements.mui_components import MuiToggleButton

from quetzal.dtos.video import QueryVideo, DatabaseVideo
from quetzal.align_frames import QueryIdx, DatabaseIdx, Match
from quetzal_app.utils.utils import format_time, get_base64
from quetzal_app.page.page_state import PageState, Page
from quetzal_app.page.video_comparison_controller import (
    Controller,
    PlaybackController,
    ObjectDetectController,
    ObjectAnnotationController,
    PLAY_IDX_KEY,
)
from pathlib import Path
import base64
import pickle

from quetzal_app.elements.image_frame_component import image_frame

PAGE_NAME = "video_comparison"

LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("../quetzal_logo_trans.png"))
LOGO_FILE = f"data:image/jpeg;base64,{get_base64(LOGO_FILE)}"
TITLE = "Quetzal"

BORDER_RADIUS = "0.8rem"
FRAME_IDX_TXT = "Frame Index: {}/{}"
PLAYBACK_TIME_TXT = "Playback Time: {}/{}"

WIDTH_ANNOTATE_DISPLAY = 512
HEIGHT_ANNOTATE_DISPLAY = 512
LINE_WIDTH_ANNOTATE_DISPLAY = 2

controller_dict: dict[str, Controller] = {
    PlaybackController.name: PlaybackController,
    ObjectDetectController.name: ObjectDetectController,
    ObjectAnnotationController.name: ObjectAnnotationController,
}


class VideoComparisonPage(Page):
    name = PAGE_NAME

    def __init__(self, root_state: PageState, to_page: list[callable]):
        self.root_state = root_state
        self.init_page_state(root_state)
        self.to_page = to_page

    def init_page_state(self, root_state: PageState) -> PageState:
        self.page_state = PageState(
            matches=None,
            controller=PlaybackController.name,
            warp=True,
            next_frame=False,
            info_anchor=None,
            annotated_frame={
                "query": None,
                "db": None,
                "idx": -1,
            },
        )

        self.page_state.update(
            {
                PlaybackController.name: PlaybackController.initState(root_state),
                ObjectDetectController.name: ObjectDetectController.initState(
                    root_state
                ),
                ObjectAnnotationController.name: ObjectAnnotationController.initState(
                    root_state
                ),
                PLAY_IDX_KEY: 0,
            }
        )
        
        return self.page_state

    def open_file_explorer(self):
        self.init_page_state(self.root_state)
        self.to_page[0]()

    def render(self):

        st.markdown(
            f"""
                <style>
                    .block-container {{ /* Removes streamlit default white spaces in the main window*/
                            padding-top: 1.5rem;
                            padding-bottom: 1rem;
                            padding-left: 3rem;
                            padding-right: 3rem;
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

        # Initialize Variable
        if self.page_state.matches is None:
            self.page_state.update(self.root_state.comparison_matches)            

        if "first_load" not in ss:
            ss.first_load = True

        TitleContent(
            page_state=self.page_state, to_file_explorer=self.open_file_explorer
        ).render()

        controller: dict[str, Controller] = {
            k: v(self.page_state) for k, v in controller_dict.items()
        }
        with st.container(border=True):
            FrameDisplay(self.page_state).render()

            ControllerOptions(self.page_state, self.root_state.torch_device).render()
            with st.container(border=True):
                controller[self.page_state.controller].render()

            ## !! no other render beyond this!! for player-control

        if ss.first_load:
            ss.first_load = False
            st.rerun()


class TitleContent:

    def __init__(self, page_state, to_file_explorer):
        self.page_state = page_state
        self.to_file_explorer = to_file_explorer

    def handle_popover_open(self, event):
        self.page_state.info_anchor = True

    def handle_popover_close(self):
        self.page_state.info_anchor = None

    def title(self):
        with mui.Stack(
            spacing=0.5,
            direction="row",
            alignItems="center",
            justifyContent="start",
            sx={"hegiht": 51}
        ):
            mui.Avatar(
                alt="Quetzal",
                src=LOGO_FILE,
                sx={"width": 48, "height": 48 }
            )
            mui.Typography(
                TITLE,
                sx={
                    "fontSize": "h4.fontSize",
                    # /* top | left and right | bottom */
                    "margin": "0.5rem 1rem 0.25rem",
                },
            )

    def render(self):
        with elements("title"):
            with mui.Grid(container=True):
                with mui.Grid(item=True, xs=4):
                    self.title()

                with mui.Grid(item=True, xs=4):
                    mui.Box(
                        mui.icon.Compare(
                            onMouseEnter=self.handle_popover_open,
                            onMouseLeave=self.handle_popover_close,
                        ),
                        display="flex",
                        justifyContent="center",
                        alignItems="end",
                        height="100%",
                    )

                with mui.Grid(item=True, xs=4):
                    mui.Box(
                        mui.Button(
                            "Back to Home",
                            variant="text",
                            startIcon=mui.icon.Home(),
                            sx={
                                "height": "min-content",
                                "padding": 0,
                            },
                            onClick=self.to_file_explorer,
                        ),
                        display="flex",
                        justifyContent="end",
                        alignItems="end",
                        height="100%",
                    )

            with mui.Popover(
                id="mouse-over-popover",
                sx={"pointerEvents": "none"},
                open=self.page_state.info_anchor,
                anchorPosition={"top": 0, "left": 0},
                anchorOrigin={"vertical": "bottom", "horizontal": "center"},
                transformOrigin={"vertical": "bottom", "horizontal": "center"},
                onClose=self.handle_popover_close,
                disableRestoreFocus=True,
            ):
                mui.Typography(
                    'Use the slider on the "Aligned Data Frame" to compare the matched images',
                    sx={"p": 1},
                )


class FrameDisplay:

    def __init__(self, page_state):
        self.page_state = page_state

    def display_frame(self, labels, images, image_urls, frame_lens, idxs, fps, 
                      bboxes_query, labels_query, bboxes_db, labels_db, detection_run):
        match self.page_state.controller:
            case ObjectAnnotationController.name:
                self.display_annotation_frame(image_urls, bboxes_query, labels_query, bboxes_db, labels_db, detection_run)
            case _:
                total_times, curr_times = [], []
                for i in range(len(frame_lens)):
                    curr_total_time, curr_show_hours = format_time(
                        frame_lens[i] / fps[i], show_hours=False, final_time=True
                    )
                    curr_time, _ = format_time(idxs[i] / fps[i], curr_show_hours)

                    total_times.append(curr_total_time)
                    curr_times.append(curr_time)

                captions = []
                for j in range(len(idxs)):
                    captions.append([FRAME_IDX_TXT.format(idxs[j], frame_lens[j]),
                                    PLAYBACK_TIME_TXT.format(curr_times[j], total_times[j])])
                    
                # keys = ["image_comparison" + str(fps[k]) for k in range(len(fps))]            
                image_frame(
                    image_urls=images,
                    captions= captions,
                    labels=labels,
                    starting_point=0,
                    dark_mode=False,
                    key="image_comparison" + str(fps[0]) + str(fps[1])
                )

    def display_annotation_frame(self, image_urls, bboxes_query, labels_query, bboxes_db, labels_db, detection_run):
        label_to_idx = lambda s : label_list.index(s)
        label_to_query = lambda s : s + "_query"
        label_to_db = lambda s : s +"_db"

        if detection_run:
            labels_query = list(map(label_to_query, labels_query))
            labels_db = list(map(label_to_db, labels_db))
            label_list = list(set(list(labels_query) + list(labels_db))) 
            labels_query = list(map(label_to_idx, labels_query))
            labels_db = list(map(label_to_idx, labels_db))
            bboxes_query = [x for x in list(bboxes_query)]
            bboxes_db = [x for x in list(bboxes_db)]
        else:
            label_list = ['deer', 'human', 'dog', 'penguin', 'framingo', 'teddy bear']            
            bboxes_query = bboxes_db = [(0.0, 0.0, 0.2857142857142857, 0.21413276231263384), (0.014285714285714285, 0.042826552462526764, 0.15714285714285714, 0.3640256959314775)]
            labels_query = labels_db = [0,3]

        [query_img, database_img] = image_urls

        result_dict = {}
        result_dict[query_img] = {'bboxes':list(bboxes_query),'labels':labels_query}
        result_dict[database_img] = {'bboxes': list(bboxes_db),'labels':labels_db}
        st.session_state['result'] = result_dict.copy()


        # # sync up frames
        # combined_bboxes = []
        # combined_labels = []

        # for i in range(len(bboxes_query)):
        #     if(bboxes_query[i] not in combined_bboxes):
        #         combined_bboxes.append(bboxes_query[i])
        #         combined_labels.append(labels_query[i])

        # for i in range(len(bboxes_db)):
        #     if(bboxes_db[i] not in combined_bboxes):
        #         combined_bboxes.append(bboxes_db[i])
        #         combined_labels.append(labels_db[i])

        # result_dict = {}
        # result_dict[query_img] = {'bboxes':list(combined_bboxes),'labels':combined_labels}
        # result_dict[database_img] = {'bboxes': list(combined_bboxes),'labels':combined_labels}
        # st.session_state['result'] = result_dict.copy()

        # map images on each side for bounding box
        # choose only the bounding box origin for segmentation

        c1, c2 = st.columns([1, 1])
        combined_bboxes= st.session_state["result"][query_img]["bboxes"] + st.session_state["result"][database_img]["bboxes"]
        combined_labels = st.session_state["result"][query_img]["labels"] + st.session_state["result"][database_img]["labels"]
        with c1:
            st.session_state.out = detection(image_path=query_img, 
                                               label_list=label_list, 
                                               bboxes=combined_bboxes,
                                               bbox_format='REL_XYXY',
                                               labels=combined_labels,
                                               metaDatas=[],
                                               image_height=512,
                                               image_width=512,
                                               line_width=1.0,
                                               ui_position='left',
                                               class_select_position=None,
                                               item_editor_position=None,
                                               item_selector_position=None,
                                               class_select_type='select',
                                               item_editor=True,
                                               item_selector=True,
                                               edit_description=False,
                                               ui_size='left',
                                               ui_left_size=None,
                                               ui_bottom_size=None,
                                               ui_right_size=None,
                                               key=None)
            print("query: ", st.session_state.out['bbox'])
            if st.session_state.out['bbox'] != None:
                # separate changes to boxes in query and database image
                # given the label of bbox, check if suffix is "_query" or "_db" and sort accordingly
                result_query = []
                for box in st.session_state.out['bbox']:
                    if box not in st.session_state['result'][database_img]['bboxes']:
                        result_query.append(box)
                    
                st.session_state['result'][query_img]['bboxes'] = result_query
                # st.session_state["result"][query_img]["labels"] = st.session_state.out['labels']

        with c2:
            st.session_state.out = detection(image_path=database_img, 
                                               label_list=label_list, 
                                               bboxes=combined_bboxes,
                                               bbox_format='REL_XYXY',
                                               labels=combined_labels,
                                               metaDatas=[],
                                               image_height=512,
                                               image_width=512,
                                               line_width=1.0,
                                               ui_position='left',
                                               class_select_position=None,
                                               item_editor_position=None,
                                               item_selector_position=None,
                                               class_select_type='select',
                                               item_editor=True,
                                               item_selector=True,
                                               edit_description=False,
                                               ui_size='left',
                                               ui_left_size=None,
                                               ui_bottom_size=None,
                                               ui_right_size=None,
                                               key=None)
            
            print("database: ", st.session_state.out['bbox'])

            if st.session_state.out['bbox'] != None:
                    result_db = []
                    for box in st.session_state.out['bbox']:
                        if box not in st.session_state['result'][query_img]['bboxes']:
                            result_db.append(box)
                    st.session_state['result'][database_img]['bboxes'] = result_db
                # st.session_state["result"][database_img]["labels"] = st.session_state.out['labels']








    def render(self):
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]
        query: QueryVideo = self.page_state.query
        database: DatabaseVideo = self.page_state.database

        bboxes_query = labels_query = bboxes_db = labels_db = []
        detection_run = False
        match self.page_state.controller:
            case PlaybackController.name if self.page_state.warp:
                query_img = self.page_state.warp_query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]
            case ObjectDetectController.name if self.page_state.annotated_frame[
                "idx"
            ] == ss.slider:
                query_img = self.page_state.annotated_frame["query"]
                database_img = self.page_state.annotated_frame["db"]
            case ObjectAnnotationController.name if self.page_state.annotated_frame[
                "idx"
            ] == ss.slider:
                query_img = self.page_state.annotated_frame["query"]
                database_img = self.page_state.annotated_frame["db"]
                bboxes_query = self.page_state.annotated_frame["bboxes_query"]
                labels_query = self.page_state.annotated_frame["labels_query"]
                bboxes_db = self.page_state.annotated_frame["bboxes_db"]
                labels_db = self.page_state.annotated_frame["labels_db"]
                detection_run = True
            case ObjectDetectController.name if self.page_state.warp:
                query_img = self.page_state.warp_query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]
            case ObjectAnnotationController.name if self.page_state.warp:
                query_img = self.page_state.warp_query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]
            case _:
                query_img = self.page_state.query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]

        query_img_base64 = f"data:image/jpeg;base64,{get_base64(query_img)}"
        db_img_base64 = f"data:image/jpeg;base64,{get_base64(database_img)}"
        image_urls = [query_img, database_img]

        labels = ["Query Frame: " + query.name,"Aligned Database Frame: " + database.name]
        images = [[query_img_base64], [query_img_base64, db_img_base64]]
        frame_lens = [len(self.page_state.query_frames), len(self.page_state.db_frames)]
        idxs = [query_idx, db_idx]
        fps = [QueryVideo.FPS, DatabaseVideo.FPS]
    
        self.display_frame(
            labels=labels, 
            images=images, 
            image_urls=image_urls,
            frame_lens=frame_lens, 
            idxs=idxs, 
            fps=fps,
            bboxes_query=bboxes_query,
            labels_query=labels_query,
            bboxes_db=bboxes_db,
            labels_db=labels_db, 
            detection_run=detection_run)
            

        


class ControllerOptions:

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

    def __init__(self, page_state, torch_device):
        self.page_state = page_state
        self.torch_device = torch_device

    def handleSwitch(self):
        self.page_state.warp = not self.page_state.warp

    def stotrePageState(self, page):
        controller_dict[page].storeState(self.page_state[page])

    def loadPageState(self, page):
        controller_dict[page].loadState(self.page_state[page])

    def handleController(self, event, controller):
        if controller is not None:
            self.stotrePageState(self.page_state.controller)
            self.loadPageState(controller)
            self.page_state.controller = controller

    def render(self):
        toggle_buttons = [
            MuiToggleButton(PlaybackController.name, "PlayArrow", "Playback Control"),
            MuiToggleButton(
                ObjectDetectController.name, "CenterFocusStrong", "Object Detection"
            ),
            MuiToggleButton(
                ObjectAnnotationController.name, "Create", "Object Annotation"
            ),
        ]

        with elements("tabs"):
            with mui.Stack(
                spacing=2,
                direction="row",
                alignItems="start",
                justifyContent="space-between",
                sx={"my": 0, "maxHeight": "calc(30.39px - 22.31px + 1rem)"},
            ):
                with mui.ToggleButtonGroup(
                    value=self.page_state.controller,
                    onChange=self.handleController,
                    exclusive=True,
                    sx=self.toggle_buttons_style,
                ):
                    for button in toggle_buttons:
                        button.render()

                with mui.Stack(
                    direction="row",
                    spacing=0,
                    alignItems="start",
                    sx={"maxHeight": "calc(30.39px - 22.31px + 1rem)"},
                ):
                    mui.Typography("Image Warp", sx={"ml": "0.3rem", "py": "7px"})
                    mui.Switch(checked=self.page_state.warp, onChange=self.handleSwitch)