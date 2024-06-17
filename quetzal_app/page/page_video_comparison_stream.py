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


PAGE_NAME = "video_comparison_stream"

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
    
    
from datetime import datetime, timezone, timedelta
import os
import requests
from bs4 import BeautifulSoup
import redis
import pandas as pd
import json

def convert_milliseconds_to_datetime(milliseconds):
        seconds = milliseconds / 1000.0
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        return dt
    
def connect_redis(host, port, username, password):
    red = redis.Redis(
        host=host,
        port=port,
        username=username,
        password=password,
        decode_responses=True,
    )   
    return red

DATA_TYPES = {
    "latitude": "float",
    "longitude": "float",
    "altitude": "float",
    "bearing": "int",
    "rssi": "int",
    "battery": "int",
    "mag": "int",
    # "sats": int,
}

def stream_to_dataframe(results, types=DATA_TYPES ) -> pd.DataFrame:
    _container = {}
    for item in results:
        _container[item[0]] = json.loads(json.dumps(item[1]))

    df = pd.DataFrame.from_dict(_container, orient='index')
    if types is not None:
        df = df.astype(types)

    return df

def fetch_and_filter_entries(redis_client, key, min_t, max_t):
    results = redis_client.xrange(key, min="-", max="+")
    processed_results = [(int(item[0].split('-')[0]), item[1]) for item in results]

    # Filter entries within the buffer range of target timestamps
    filtered_results = [
        entry for entry in processed_results
        if min_t <= entry[0] and entry[0] <= max_t
    ]
    print(len(filtered_results))
    
    return filtered_results
    
def pair_images_with_data(images, redis_entries):
    """
    Pair each image with the closest matching data based on timestamp.

    :param images: List of image info tuples (path, timestamp string, timestamp ms).
    :param redis_entries: List of filtered Redis entries.
    :return: List of tuples (image_path, closest_data).
    """
    paired_list = []
    for img_path, _, img_timestamp in images:
        closest_data = min(redis_entries, key=lambda x: abs(x[0] - img_timestamp))
        paired_list.append((img_path, closest_data[1]))  # Assuming you want the data part of the entry
    return paired_list    

class CloudletMonitorThread(threading.Thread):

    def __init__(self, 
        url, 
        save_dir, 
        pipline_executor: Pipeline, 
        redis_host,
        redis_port,
        redis_username,
        redis_password,
        redis_key = "telemetry.harpyeagle",
        poll_interval=1.0, 
        fps = 2,
        last_frame_mode = False,
    ):
        # Last_frame_mode = only load last frames
        threading.Thread.__init__(self)
        self.url = url
        self.save_dir = save_dir
        self.fps = fps
        self.poll_interval = poll_interval
        self.last_known_state = set()
        self.reference_time = None
        self.pipeline_executor = pipline_executor
        self.stop_event = threading.Event()
        self.logger = logging.getLogger("CloudletMonitorThread")
        self.logger.setLevel(logging.DEBUG)
        self.frame_num = 1
        self.logger.debug("Initializing with: " + str(url))
        self.redis = connect_redis(redis_host,redis_port, redis_username, redis_password)
        self.redis_key = redis_key
        self.last_frame_mode = last_frame_mode

    def run(self):
        self.logger.debug("Start Running")
        ss.CloudletMonitorThreadLife = True
        while not self.stop_event.is_set():
            try:
                if ss.CloudletMonitorThreadLife: # session state disapear when session closes
                    self.scan_cloudlet()
                    time.sleep(self.poll_interval)
                else:
                    self.logger.debug("Should never reach here")
                    break
            except Exception as e:
                print(e)
                self.logger.debug("Session Closed")
                self.pipeline_executor.end()
                break
            
        self.logger.debug("Stopped")
        
    def scan_cloudlet(self):
        delta_t = 1 / self.fps
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(('.jpg', '.png', '.jpeg'))]
        img_list = []
        current_state = {image for image in image_links}
        added = current_state - self.last_known_state        
        
        if added:
            filtered_frames = []
            added_list = [file for file in added]
            added_list.sort()
            
            if self.last_frame_mode:
                last_img_link = added_list[-1]
                img_name = last_img_link.split('/')[-1]
                timestamp_ms = int(img_name.split('.')[0])
                timestamp_datetime = convert_milliseconds_to_datetime(timestamp_ms)
                self.reference_time = timestamp_datetime - timedelta(self.poll_interval + delta_t / 2)
                
            for img_link in image_links:
                if not img_link.startswith('http'):
                    img_link = self.url + img_link
                img_name = img_link.split('/')[-1]
                timestamp_ms = int(img_name.split('.')[0])
                timestamp_datetime = convert_milliseconds_to_datetime(timestamp_ms)
                
                if self.reference_time is None:
                    self.reference_time = timestamp_datetime  # Set the first frame as the reference
                    filtered_frames.append((img_link, timestamp_datetime, timestamp_ms))
                else:
                    # Calculate the difference from the last reference time in seconds
                    time_difference = (timestamp_datetime - self.reference_time).total_seconds()
                    if time_difference >= delta_t:
                        # Update the reference time and add the frame to the list
                        self.reference_time += timedelta(seconds=delta_t)
                        # reference_time = timestamp_datetime
                        filtered_frames.append((img_link, timestamp_datetime, timestamp_ms))
            
            for img_link, timestamp_datetime, timestamp_ms in filtered_frames:
                img_name = os.path.basename(img_link)
                img_path = os.path.join(self.save_dir, f"frame{self.frame_num:05}.jpg")
                time_stemp_str = timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC'
                img_list.append((img_path, time_stemp_str, timestamp_ms))

                # Optional: Download and save the filtered image
                img_response = requests.get(img_link)
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                
                self.frame_num += 1
                        
            print("here1")
            image_redis_ids  = [value[2] for value in img_list]

            min_id = min(image_redis_ids) # Adding '-0' to format as Redis stream ID
            max_id = max(image_redis_ids)
                        
            # for k in self.redis.keys(self.redis_key):
            filtered_entries = fetch_and_filter_entries(self.redis, f"{self.redis_key}", min_id, max_id)
            if filtered_entries:
                paired_data = pair_images_with_data(img_list, filtered_entries)
                paired_data = [(img_path, (float(data['latitude']), float(data['longitude']))) for img_path, data in paired_data]
                self.pipeline_executor.submit(paired_data)
        
        self.last_known_state = current_state
        
    def stop(self):
        self.stop_event.set()

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
            added_list = [((file, convert_path(file, 512)), self.gps[i]) for i, file in enumerate(added_list)]
            
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
        self.page_state.query_frames = self.query_frames
        self.page_state.warp_query_frames = self.query_frames
        self.page_state[PLAY_IDX_KEY] = len(self.matches) - 1
        
        # ss[SLIDER_KEY] = len(self.matches) - 1
        # ss[SLIDER_KEY + "value"] = ss[SLIDER_KEY]
        
        frontend_rerun()
    
    def stop(self):
        self.stop_event.set()
        
class VideoComparisonStreamPage(Page):
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
            # query_video: Video = self.page_state.query
            db_video: Video = self.page_state.database
            db_gps_anafi = AnafiGPS(db_video)
            # query_gps_anfi = AnafiGPS(query_video)
            # query_gps_look_at = query_gps_anfi.get_look_at_gps(camera_angle=45)
            
            # query_video.resolution = 256
            
            def realtime_align_engine_wrapper():
                engine: AbstractEngine = RealtimeAlignmentEngine(
                    device=self.root_state.torch_device, 
                    database_video=db_video,
                    database_gps=db_gps_anafi,
                    gps_look_at=False,
                    query_num=5
                )
                return engine
            
            pipeline = Pipeline([
                    (realtime_align_engine_wrapper, 1, None),
                ], 
                queue_maxsize=2000,
                verbose = False
            )
            
            ss.pipeline_thread = pipeline
            
            monitor_thread = CloudletMonitorThread(
                url = "",
                save_dir = "",
                pipline_executor=pipeline,
                poll_interval=1,
                redis_host="",
                redis_port="",
                redis_username="",
                redis_password="",
                fps=2,
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
