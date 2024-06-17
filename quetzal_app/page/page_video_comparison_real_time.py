import streamlit as st
import os
from streamlit import session_state as ss

from streamlit_elements import elements, mui
from quetzal_app.elements.mui_components import MuiToggleButton

from quetzal.dtos.video import QueryVideo, DatabaseVideo, Video, convert_path
from quetzal.align_frames import QueryIdx, DatabaseIdx, Match
from quetzal_app.utils.utils import format_time, get_base64
from quetzal_app.page.page_state import PageState, Page
from quetzal_app.page.video_comparison_controller import (
    Controller,
    PlaybackController,
    ObjectDetectController,
    PLAY_IDX_KEY,
)
from pathlib import Path

from quetzal_app.elements.image_frame_component import image_frame
from quetzal_app.notifier import get_browser_session_id, get_streamlit_session
from streamlit.runtime.scriptrunner import add_script_run_ctx
from quetzal.engines.pipeline_executor import Pipeline
from quetzal.engines.engine import AbstractEngine
from quetzal.engines.align_engine.realtime_engine import RealtimeAlignmentEngine
from quetzal.dtos.gps import AnafiGPS, AbstractGPS

import threading
import queue
import os
import time
import logging


PAGE_NAME = "video_comparison_real_time"

LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("../quetzal_logo_trans.png"))
LOGO_FILE = f"data:image/jpeg;base64,{get_base64(LOGO_FILE)}"
TITLE = "Quetzal"

BORDER_RADIUS = "0.8rem"
FRAME_IDX_TXT = "Frame Index: {}/{}"
PLAYBACK_TIME_TXT = "Playback Time: {}/{}"

controller_dict: dict[str, Controller] = {
    PlaybackController.name: PlaybackController,
    ObjectDetectController.name: ObjectDetectController,
}

logging.basicConfig()

from glob import glob

streamlit_session = get_streamlit_session(get_browser_session_id())

def frontend_rerun() -> None:
    streamlit_session._handle_rerun_script_request()

class DirectoryMonitorThread(threading.Thread):

    def __init__(self, directory, pipline_executor: Pipeline, poll_interval=1.0, gps=None):
        threading.Thread.__init__(self)
        self.dir = directory
        self.poll_interval = poll_interval
        self.last_known_state = {}
        self.pipeline_executor = pipline_executor
        self.stop_event = threading.Event()
        self.logger = logging.getLogger("DirectoryMonitorThread")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.debug("Initializing with: " + str(directory))
        self.gps = gps

    def run(self):
        self.logger.debug("Start Running")
        ss.directoryMonitorThreadLife = True
        while not self.stop_event.is_set():
            try:
                if ss.directoryMonitorThreadLife: # session state disapear when session closes
                    self.scan_directory()
                    time.sleep(self.poll_interval)
                else:
                    self.logger.debug("Should never reach here")
                    break
            except:
                self.logger.debug("Session Closed")
                self.pipeline_executor.end()
                break
            
        self.logger.debug("Stopped")

    def scan_directory(self):
        files = glob(os.path.join(self.dir, f"frame*.jpg"))
        current_state = {f: os.path.getmtime(f) for f in files if os.path.isfile(f)}
        added = current_state.keys() - self.last_known_state.keys()
        removed = self.last_known_state.keys() - current_state.keys()
        modified = {f for f in current_state if f in self.last_known_state and current_state[f] != self.last_known_state[f]}
        
        if added:
            added_list = [file for file in added]
            added_list.sort()
            # added_list = [((file, convert_path(file, 512)), self.gps[i]) for i, file in enumerate(added_list)]
            added_list = [(file, self.gps[i]) for i, file in enumerate(added_list)]

            
            self.pipeline_executor.submit(added_list)
        if removed:
            pass
        if modified:
            pass

        self.last_known_state = current_state
    
    def stop(self):
        self.stop_event.set()

from quetzal_app.page.video_comparison_controller import SLIDER_KEY

class MatchUpdateThread(threading.Thread):

    def __init__(self, page_state, pipline_executor: Pipeline, timeout=60, wait_interval=5):
        threading.Thread.__init__(self)
        self.page_state = page_state
        self.last_known_state = {}
        self.pipeline_executor = pipline_executor
        self.stop_event = threading.Event()
        self.matches = list()
        self.logger = logging.getLogger("MatchUpdateThread")
        self.logger.setLevel(logging.DEBUG)
        self.waited = 0
        self.wait_interval = wait_interval
        self.timeout = timeout
        self.query_frames = list()

    def run(self):
        self.logger.debug("Start Running")
        ss.MatchUpdateThread = True
    
        while not self.stop_event.is_set():
            try:
                if ss.MatchUpdateThread: # session state disapear when session closes
                    rv = self.pipeline_executor.get_result()
                    if rv is None:
                        time.sleep(self.wait_interval)
                        self.waited += self.wait_interval
                    
                    else:
                        self.waited = 0
                        query_idx, db_idx, query_frame = rv
                        self.matches.append((query_idx, db_idx))
                        self.query_frames.append(query_frame)
                        self.update_matches()
                else:
                    self.logger.debug("Should never reach here")
                    break
                
                if self.waited >= self.timeout:
                    self.logger.debug("Input Inactive")
                    break
            except:
                self.logger.debug("Session Closed")
                break
            
        self.logger.debug("Stopped")

    def update_matches(self):
        self.page_state.matches = self.matches
        self.page_state[PLAY_IDX_KEY] = len(self.matches) - 1
        
        # ss[SLIDER_KEY] = len(self.matches) - 1
        # ss[SLIDER_KEY + "value"] = ss[SLIDER_KEY]
        
        frontend_rerun()
    
    def stop(self):
        self.stop_event.set()
        
class VideoComparisonRealTimePage(Page):
    name = PAGE_NAME

    def __init__(self, root_state: PageState, to_page: list[callable]):
        self.root_state = root_state
        self.page_state = self.init_page_state(root_state)
        self.to_page = to_page
        

    def init_page_state(self, root_state: PageState) -> PageState:
        init_state = PageState(
            matches=None,
            controller=PlaybackController.name,
            warp=False,
            next_frame=False,
            info_anchor=None,
            annotated_frame={
                "query": None,
                "db": None,
                "idx": -1,
            },
        )

        init_state.update(
            {
                PlaybackController.name: PlaybackController.initState(root_state),
                # ObjectDetectController.name: ObjectDetectController.initState(
                #     root_state
                # ),
                PLAY_IDX_KEY: 0,
            }
        )

        return init_state

    def open_file_explorer(self):
        ss.first_load = True
        self.init_page_state(self.root_state)
        
        pipeline: Pipeline = ss.pipeline_thread
        pipeline.end()
        ss.pipeline_thread = None
        
        monitor_thread: DirectoryMonitorThread = ss.monitor_thread
        monitor_thread.stop()
        monitor_thread.join()
        ss.monitor_thread = None
        
        update_thread: MatchUpdateThread = ss.update_thread
        update_thread.stop()
        update_thread.join()
        ss.update_thread = None
        
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
        if self.page_state.matches == None:
            self.page_state.update(self.root_state.comparison_matches)

        if "first_load" not in ss or ss.first_load:
            query_video: Video = self.page_state.query
            db_video: Video = self.page_state.database
            db_gps_anafi = AnafiGPS(db_video)
            query_gps_anfi = AnafiGPS(query_video)
            query_gps_look_at = query_gps_anfi.get_look_at_gps(camera_angle=45)
            
            query_video.resolution = 1024
            
            def realtime_align_engine_wrapper():
                engine: AbstractEngine = RealtimeAlignmentEngine(
                    device=self.root_state.torch_device, 
                    database_video=db_video,
                    database_gps=db_gps_anafi,
                    query_num=5,
                )
                return engine
            
            pipeline = Pipeline([
                    (realtime_align_engine_wrapper, 1, None),
                ], 
                queue_maxsize=2000,
                verbose = False
            )
            
            ss.pipeline_thread = pipeline
            
            monitor_thread = DirectoryMonitorThread(
                query_video.dataset_dir, 
                pipline_executor=pipeline,
                poll_interval=1,
                gps=query_gps_look_at
            )
            add_script_run_ctx(monitor_thread)
            ss.monitor_thread = monitor_thread
            
            match_update_thread = MatchUpdateThread(
                page_state=self.page_state,
                pipline_executor=pipeline
            )
            add_script_run_ctx(match_update_thread)
            ss.update_thread = match_update_thread
            
            pipeline.start()
            monitor_thread.start()
            match_update_thread.start()

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

    # def display_frame(self, label, images, frame_len, idx, fps):
    #     total_time, show_hours = format_time(
    #         frame_len / fps, show_hours=False, final_time=True
    #     )
    #     curr_time, _ = format_time(idx / fps, show_hours)

    #     image_frame(
    #         image_urls=images,
    #         captions=[
    #             FRAME_IDX_TXT.format(idx, frame_len),
    #             PLAYBACK_TIME_TXT.format(curr_time, total_time),
    #         ],
    #         label=label,
    #         starting_point=0,
    #         dark_mode=False,
    #         key="image_comparison" + str(fps),
    #     )

    # def render(self):
    #     match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
    #     query_idx: QueryIdx = match[0]
    #     db_idx: DatabaseIdx = match[1]
    #     query: QueryVideo = self.page_state.query
    #     database: DatabaseVideo = self.page_state.database

    #     match self.page_state.controller:
    #         case PlaybackController.name if self.page_state.warp:
    #             query_img = self.page_state.warp_query_frames[query_idx]
    #             database_img = self.page_state.db_frames[db_idx]
    #         case ObjectDetectController.name if self.page_state.annotated_frame[
    #             "idx"
    #         ] == ss.slider:
    #             query_img = self.page_state.annotated_frame["query"]
    #             database_img = self.page_state.annotated_frame["db"]
    #         case _:
    #             query_img = self.page_state.query_frames[query_idx]
    #             database_img = self.page_state.db_frames[db_idx]

    #     query_img_base64 = f"data:image/jpeg;base64,{get_base64(query_img)}"
    #     db_img_base64 = f"data:image/jpeg;base64,{get_base64(database_img)}"

    #     imgc1, imgc2 = st.columns([1, 1], gap="small")
    #     with imgc1:
    #         self.display_frame(
    #             label="Query Frame: " + query.name,
    #             images=[query_img_base64],
    #             frame_len=len(self.page_state.query_frames),
    #             idx=query_idx,
    #             fps=QueryVideo.FPS,
    #         )

    #     with imgc2:
    #         self.display_frame(
    #             label="Aligned Database Frame: " + database.name,
    #             images=[query_img_base64, db_img_base64],
    #             frame_len=len(self.page_state.db_frames),
    #             idx=db_idx,
    #             fps=DatabaseVideo.FPS,
    #         )
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
            # MuiToggleButton(
            #     ObjectDetectController.name, "CenterFocusStrong", "Object Detection"
            # ),
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
