# from quetzal_app.page.page_video_comparison import VideoComparisonPage
# from quetzal_app.page.page_file_explorer import FileExplorerPage
# from quetzal_app.page.page_state import AppState, PageState, Page

# import streamlit as st
# from streamlit import session_state as ss
# import argparse
# import torch

# from threading import Lock
# from streamlit.web.server.websocket_headers import _get_websocket_headers
# import os
# from pathlib import Path

# LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("quetzal_logo_trans.png"))


# dataset_layout_help = """
#     Dataset structure:
#     root_datasets_dir/
#     |
#     ├──user_name1/
#     |   |
#     |   ├── project_name1/
#     |   |   ├── video_name1.mp4
#     |   |   ├── video_name2.mp4
#     |   |   ├── ...
#     |   |   |
#     |   |   ├── subproject_name/
#     |   |   |   ├──video_name1.mp4
#     |   |   |   └── ...
#     |   |   └── ...
#     |   |
#     |   |
#     |   └── project_name2/
#     |
#     ├──user_name2/
#     └── ...

#     metadata_directory/

#     ├── user_name1.info.txt
#     ├── user_name1.meta.txt
#     ├── user_name1/
#     |   |
#     |   ├── project_name1.info.txt
#     |   ├── project_name1.meta.txt
#     |   ├── project_name1/
#     |   |   ├── video_name1.mp4.info.txt
#     |   |   ├── video_name1.mp4.meta.txt
#     |   |   |
#     |   |   ├── video_name2.mp4.info.txt
#     |   |   ├── video_name2.mp4.meta.txt
#     |   |   |
#     |   |   ├── ...
#     |   |   |
#     |   |   ├── database/
#     |   |   |   ├── video_name1/
#     |   |   |   |   ├── frames_{fps}_{resolution}/
#     |   |   |   |   |   ├── frame_%05d.jpg
#     |   |   |   |   |   └── ...
#     |   |   |   |   └── ...
#     |   |   |   └── video_name2/
#     |   |   |       ├── frames_{fps}_{resolution}/
#     |   |   |       |   ├── frame_%05d.jpg
#     |   |   |       |   └── ...
#     |   |   |       └── ...
#     |   |   |
#     |   |   ├── query/
#     |   |   |   ├── video_name2/
#     |   |   |   |   ├── frames_{fps}_{resolution}/
#     |   |   |   |   |   ├── frame_%05d.jpg
#     |   |   |   |   |   └── ...
#     |   |   |   |   └── ...
#     |   |   |   └── ...
#     |   |   |
#     |   |   ├── subproject_name.info.txt
#     |   |   ├── subproject_name.meta.txt
#     |   |   ├── subproject_name/
#     |   |   |   ├──video_name1.mp4.info.txt
#     |   |   |   ├──video_name1.mp4.meta.txt
#     |   |   |   └── ...
#     |   |   └── ...
#     |   |
#     |   |
#     |   └── project_name2/
#     |
#     ├── user_name1.info.txt
#     ├── user_name1.meta.txt
#     ├── user_name2/
#     └── ...
#         """

# st.set_page_config(layout="wide", page_title="Quetzal", page_icon=LOGO_FILE)

# @st.cache_data
# def parse_args():
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.RawTextHelpFormatter,
#         description="This program aligns Two Videos",
#         epilog=dataset_layout_help,
#     )

#     parser.add_argument(
#         "-d",
#         "--dataset-root",
#         default="./data/home/root",
#         help="Root directory of datasets",
#     )
#     parser.add_argument(
#         "-m",
#         "--metadata-root",
#         default="./data/meta_data/root",
#         help="Meta data directory of datasets",
#     )
#     parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=True)
#     parser.add_argument("--cuda-device", help="Select cuda device", default=0, type=int)
#     parser.add_argument("-u", "--user", default="default_user")
#     args = parser.parse_args()

#     dataset_root = args.dataset_root
#     meta_data_root = args.metadata_root

#     available_gpus = torch.cuda.device_count()
#     print(f"Avaliable GPU={available_gpus}")
#     if args.cuda and available_gpus > 0:
#         cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
#         torch_device = torch.device("cuda:" + str(cuda_device))
#     else:
#         torch_device = torch.device("cpu")

#     print(torch_device)

#     return dataset_root, meta_data_root, cuda_device, torch_device, args.user


# dataset_root, meta_data_root, cuda_device, torch_device, user = parse_args()
# headers = _get_websocket_headers()
# user = headers.get("X-Forwarded-User", user)

# page_list: list[Page] = [FileExplorerPage, VideoComparisonPage]
# page_dict: dict[str, Page] = {page.name: page for page in page_list}

# if "page_states" not in ss:
#     app_state = AppState()
#     root_state = PageState(
#         root_dir=dataset_root,
#         metadata_dir=meta_data_root,
#         cuda_device=cuda_device,
#         torch_device=torch_device,
#         page=FileExplorerPage.name,
#         user=user,
#         comparison_matches=None,
#     )

#     root_state.page = FileExplorerPage.name

#     def build_to_page(page: Page):
#         def to_page():
#             root_state.page = page.name
#             print("to_page", page.name)

#         return to_page

#     to_page = [build_to_page(page) for page in page_list]

#     ss.pages = dict()

#     app_state.root = root_state
#     for key, page in page_dict.items():
#         page_object = page(root_state=root_state, to_page=to_page)
#         ss.pages[key] = page_object
#         app_state[key] = page_object.page_state

#     ss.page_states = app_state
#     ss.lock = Lock()

# ss.pages[ss.page_states.root.page].render()


import streamlit as st
from quetzal_app.notifier import get_browser_session_id, get_streamlit_session
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

import threading
import queue
import os
import time

class Event:
    def __init__(self, event_type, data=None):
        self.event_type = event_type
        self.data = data

class Observer:
    def on_event(self, event):
        raise NotImplementedError
    
class WorkerThread(threading.Thread, Observer):
    def __init__(self, thread_id, event_queue):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.event_queue = event_queue

    def on_event(self, event):
        print(f"Thread {self.thread_id} processed event: {event.event_type}, Data: {event.data}")

    def run(self):
        while True:
            try:
                event = self.event_queue.get(timeout=1)
                self.on_event(event)
            except queue.Empty:
                continue

class DirectoryMonitorThread(threading.Thread):
    def __init__(self, directory, event_queue, poll_interval=1.0):
        threading.Thread.__init__(self)
        self.directory = directory
        self.event_queue = event_queue
        self.poll_interval = poll_interval
        self.last_known_state = {}

    def run(self):
        while True:
            self.scan_directory()
            time.sleep(self.poll_interval)

    def scan_directory(self):
        current_state = {f: os.path.getmtime(f) for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))}
        added = current_state.keys() - self.last_known_state.keys()
        removed = self.last_known_state.keys() - current_state.keys()
        modified = {f for f in current_state if f in self.last_known_state and current_state[f] != self.last_known_state[f]}

        if added:
            self.event_queue.put(Event("file_added", list(added)))
        if removed:
            self.event_queue.put(Event("file_removed", list(removed)))
        if modified:
            self.event_queue.put(Event("file_modified", list(modified)))

        self.last_known_state = current_state

def main():
    event_queue = queue.Queue()
    directory_monitor = DirectoryMonitorThread("./watched_directory", event_queue)
    worker = WorkerThread(1, event_queue)

    directory_monitor.start()
    worker.start()

    directory_monitor.join()
    worker.join()

if __name__ == "__main__":
    main()


streamlit_session = get_streamlit_session(get_browser_session_id())
       

def notify() -> None:
    streamlit_session._handle_rerun_script_request()

def delay_notify():        
    while True:
        if not st.session_state.playback:
            # print("killing Thread")
            break
        
        time.sleep(1)
        
        if not st.session_state.playback:
            # print("killing Thread")
            break
        
        st.session_state.counter += 1
        # print("notify!")
        notify()
        

st.title("Hello World")

if "playback" not in st.session_state:
    st.session_state.playback = False

if "counter" not in st.session_state:
    st.session_state.counter = 0
    
st.write(f"Counter: {st.session_state.counter}")

if st.button("Notify"):
    st.session_state.playback = not st.session_state.playback
    
    # print("clikced!!")
    if st.session_state.playback:
        # print("new Thread")
        t = threading.Thread(target=delay_notify)
        add_script_run_ctx(t)
        # st.session_state.closer = SessionCloser(t)
        t.start()
        st.session_state.thread = t
        
    else:
        # print("try to join")
        t: threading.Thread = st.session_state.thread
        t.join()
        # del st.session_state.closer

import asyncio
import threading
import gc

loops = []
# for obj in gc.get_objects():
#     if isinstance(obj, SessionCloser):
#         print(obj)
#     try:
#         if isinstance(obj, asyncio.BaseEventLoop):
#             loops.append(obj)
#     except ReferenceError:
#         ...
    
# main_thread = [t for t in threading.enumerate()]
# print(main_thread)
# print(loops)
# print(streamlit_session._state)

