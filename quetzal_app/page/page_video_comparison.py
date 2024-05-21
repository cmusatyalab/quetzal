import streamlit as st
import os
from streamlit import session_state as ss

from streamlit_elements import elements, mui
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

from quetzal_app.elements.image_frame_component import image_frame

PAGE_NAME = "video_comparison"

LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("../quetzal_logo_trans.png"))
LOGO_FILE = f"data:image/jpeg;base64,{get_base64(LOGO_FILE)}"
TITLE = "Quetzal"

BORDER_RADIUS = "0.8rem"
FRAME_IDX_TXT = "Frame Index: {}/{}"
PLAYBACK_TIME_TXT = "Playback Time: {}/{}"

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

    def display_frame(self, labels, images, frame_lens, idxs, fps):
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

    def render(self):
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]
        query: QueryVideo = self.page_state.query
        database: DatabaseVideo = self.page_state.database

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
            case _:
                query_img = self.page_state.query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]

        query_img_base64 = f"data:image/jpeg;base64,{get_base64(query_img)}"
        db_img_base64 = f"data:image/jpeg;base64,{get_base64(database_img)}"

        labels = ["Query Frame: " + query.name,"Aligned Database Frame: " + database.name]
        images = [[query_img_base64], [query_img_base64, db_img_base64]]
        frame_lens = [len(self.page_state.query_frames), len(self.page_state.db_frames)]
        idxs = [query_idx, db_idx]
        fps = [QueryVideo.FPS, DatabaseVideo.FPS]

        with st.empty():
            self.display_frame(
                labels=labels, 
                images=images, 
                frame_lens=frame_lens, 
                idxs=idxs, 
                fps=fps)


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
                ObjectAnnotationController.name, "CenterFocusStrong", "Object Annotation"
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