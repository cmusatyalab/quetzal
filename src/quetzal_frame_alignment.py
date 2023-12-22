import gradio as gr
from src.video import *
import logging
from copy import deepcopy
from src.engines.vpr_engine.anyloc_engine import AnyLocEngine
from src.compute_vlad import generate_VLAD
from src.align_frames import align_video_frames, align_frame_pairs
from src.engines.detection_engine.grounding_sam_engine import GoundingSAMEngine
import supervision as sv

# from src.utils.dtw_vlad import *
from tqdm import tqdm
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import src.utils.file_tools as ft
from glob import glob
import argparse

logging.basicConfig()
logger = logging.getLogger("Main Process")
logger.setLevel(logging.DEBUG)

icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")


verbose = True
SELECT_ROUTE_MSG = "select route"
NO_CHOICE_MSG = "no choice"

dataset_layout_help = """
    Dataset structure:
    root_datasets_dir/
    |
    ├── route_name/
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

example_routes = [
    "example_mil19",
    "example_army_demo",
    "example_purdue",
    "example_hot_metal_bridge",
]
footage_domains = ["aerial", "indoor", "urban"]
grounding_sam = None


## Initialize System
def is_route_example(route_name):
    if route_name in example_routes:
        return True
    return False


def format_time(seconds, show_hours=True, final_time=False):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60

    if final_time:
        if hours == 0:
            return f"{minutes:02d}:{seconds:05.2f}", False
    else:
        if not show_hours:
            return f"{minutes:02d}:{seconds:05.2f}", False

    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}", True


def non_example_routes():
    return [
        file for file in ft.get_directories(dataset_root) if not is_route_example(file)
    ]


WELCOME_DELETE = """
**Deletion Process:**

1. **Select Files for Deletion:** 
   - Pick the route and its associated video files you wish to delete.

2. **Confirmation:**
   - Enter the designated phrase in the confirmation textbox to verify your intention to delete.

3. **Execute Deletion:**
   - After confirmation, press the "Delete" button to permanently remove the chosen files.

"""


def delete_videos(route, video, confirm: str):
    if confirm != "confirm deletion":
        choice = [
            file
            for file in ft.get_directories(dataset_root)
            if not is_route_example(file)
        ]
        return (
            gr.update(choices=choice),
            gr.update(value=""),
            "**Error: The confirmation string does not match**",
        )

    msg = ""

    delete_dir = os.path.join(dataset_root, route)
    video_file = os.path.join(delete_dir, "raw_video", video)
    video_meta = os.path.join(delete_dir, "raw_video", os.path.splitext(video)[0] + ".txt")
    database_folder = glob(
        os.path.join(delete_dir, "database", os.path.splitext(video)[0])
    )
    query_folder = glob(os.path.join(delete_dir, "query", os.path.splitext(video)[0]))

    if os.path.isfile(video_file):
        os.remove(video_file)
        if os.path.isfile(video_meta):
            os.remove(video_file)

    if database_folder and os.path.isdir(database_folder[0]):
        ft.delete_directory(database_folder[0])

    if query_folder and os.path.isdir(query_folder[0]):
        ft.delete_directory(query_folder[0])

    if ft.is_directory_effectively_empty(os.path.join(delete_dir, "raw_video")):
        ft.delete_directory(delete_dir)

    msg += f"**Removed {video}!**"
    choice = [
        file for file in ft.get_directories(dataset_root) if not is_route_example(file)
    ]
    return gr.update(choices=choice), gr.update(value=""), msg


def get_video_files_for_route(route):
    if route and route != "str":
        target_dir = os.path.join(dataset_root, route, "raw_video")
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            file_list = [
                os.path.basename(file) for file in glob(os.path.join(target_dir, "*"))
            ]
            file_list = [file for file in file_list if not file.endswith(".txt")]
            return gr.update(choices=file_list)

    return gr.update(choices=[NO_CHOICE_MSG])

def delete_tab():
    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                gr.Markdown("# Delete Uploaded Files")
            with gr.Row():
                instruction = gr.Markdown(WELCOME_DELETE, label="Instructions")
            with gr.Row():
                progess_title = gr.Markdown("## Progress")
            with gr.Row():
                progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")

        with gr.Column(scale=4):
            with gr.Row():
                route_name_input = gr.Dropdown(
                    choices=[
                        file
                        for file in ft.get_directories(dataset_root)
                        if not is_route_example(file)
                    ],
                    label="Route",
                    interactive=True,
                    allow_custom_value=False,
                    value="str",
                )
                video_name = gr.Dropdown(
                    choices=[SELECT_ROUTE_MSG],
                    label="Videos",
                    allow_custom_value=False,
                    interactive=True,
                    value="str",
                )
            with gr.Row():
                delete_confirm = gr.Textbox(
                    interactive=True, label="Type 'confirm deletion', case sensitive"
                )
            with gr.Row():
                delete_btn = gr.Button("Delete")

    route_name_input.change(
        get_video_files_for_route,
        inputs=route_name_input,
        outputs=video_name,
        show_progress=False,
    )

    route_name_input.change(
        lambda x: gr.update(choices=non_example_routes()),
        inputs = route_name_input,
        outputs= route_name_input,
        show_progress=False,
    )
    delete_btn.click(
        delete_videos,
        inputs=[route_name_input, video_name, delete_confirm],
        outputs=[route_name_input, delete_confirm, progress],
    )


WELCOME_ANALYZE = """
Please follow the steps below to start your video analysis:

1. **Select the Route:** 
   - Choose the route you wish to analyze.

2. **Choose Your Video File(s):**
   - Select the target video from within the chosen route.

3. **Select Video Type (Query or Database):**
   - Choose 'Query' for the primary video against which alignment will be performed.
   - Choose 'Database' for videos that will provide frames to be aligned to the query video.
   - Note: You need at least one video of each type for alignment results.
   - A video can be registered as both Query and Database, but this requires separate submissions.

4. **Initiate Analysis:**
   - Press the **"Analyze"** button to begin the analysis process.

Once done, head over to the **"Alignment Results"** tab to see how the database video frames align with your query video!
"""


def analyze_video(
    route_name,
    video_name,
    video_type: Literal["query", "database"],
    progress=gr.Progress(track_tqdm=True),
):
    if route_name == "str":
        return "**ERROR: Choose your route and video!**"

    if video_name == SELECT_ROUTE_MSG or video_name == "str":
        return "**ERROR: Choose your video!**"

    query_video = None
    db_video = None
    if video_type == "query":
        query_video = QueryVideo(dataset_root, route_name, video_name)
        if os.path.isfile(f"{query_video.get_dataset_dir()}/vlads.npy"):
            return 'The video has already been analyzed as "Query"'
    if video_type == "database":
        db_video = DatabaseVideo(dataset_root, route_name, video_name)
        if os.path.isfile(f"{db_video.get_dataset_dir()}/vlads.npy"):
            return 'The video has already been analyzed as "Database"'

    global grounding_sam
    if grounding_sam:
        del grounding_sam
        grounding_sam = None

    generate_VLAD(db_video, query_video, torch_device)

    return "**Success!**"


def analyze_tab():
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                gr.Markdown("# Analyze Videos")
            with gr.Row():
                instruction = gr.Markdown(WELCOME_ANALYZE, label="Instructions")

        with gr.Column(scale=6):
            with gr.Row():
                route_name_input = gr.Dropdown(
                    choices=non_example_routes(),
                    label="Route",
                    interactive=True,
                    allow_custom_value=False,
                    value="str",
                )
                video_name = gr.Dropdown(
                    choices=[SELECT_ROUTE_MSG],
                    label="Videos",
                    allow_custom_value=False,
                    interactive=True,
                    value="str",
                )
            with gr.Row():
                route_info = gr.TextArea(
                    label="Route Infomation",
                    value="Choose Route",
                    interactive=False
                    )
                video_info = gr.TextArea(
                    label="Video information",
                    value="Choose Video",
                    interactive=False
                )
            with gr.Row():
                video_type_input = gr.Radio(
                    choices=["database", "query"],
                    label="Video Type",
                    value="database",
                    interactive=True,
                )
            with gr.Row():
                analyze_btn = gr.Button("Analyze")

    with gr.Row():
        progess_title = gr.Markdown("## Progress")
    with gr.Row():
        progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")

    route_name_input.change(
        get_video_files_for_route,
        inputs=route_name_input,
        outputs=[video_name],
        show_progress=False,
    )

    route_name_input.change(
        lambda x: gr.update(choices=non_example_routes()),
        inputs = route_name_input,
        outputs= route_name_input,
        show_progress=False,
    )

    route_name_input.change(
        read_route_meta,
        inputs=route_name_input,
        outputs=route_info,
        show_progress=False
    )

    video_name.change(
        read_video_meta,
        inputs=[route_name_input, video_name],
        outputs=video_info
    )

    analyze_btn.click(
        analyze_video,
        inputs=[route_name_input, video_name, video_type_input],
        outputs=[progress],
    )

WELCOME_UPLOAD = """
**Welcome!** You can check pre-processed examples in **"Alignment Results"** Tab or analyze your own video footages.

To begin, please follow the steps below:

1. **Select the Route:** 
   - You can upload a new video for an existing route (Choose From the dropdown)
   - Or you can add a new route entirely (type in new route-name)
   - Note: Modification is restricted for the pre-defined 'example' routes*. 

2. **Choose Your Video File:**
   - Drag and drop or select click to upload your video (only one).
   - Do not touch the interface while the files are being uploaded.

2. **Update/Enter Route and Video Information(s):**
   - For future reference, add describtion for the route and the video you are uploading.

3. **Upload:**
   - Click on the **"Upload"** button to fully upload your file into the system.

**Next:** go to **"Analyze Video"** tab to run analysis on the uploaded videos or **"Delete Files"** tab to delete your files!

"""


def is_valid_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False

    ret, frame = cap.read()
    cap.release()
    return ret

def write_route_meta(metadata, route_name):
    route_dir = os.path.join(dataset_root, route_name)
    meta_file = os.path.join(route_dir, "metadata.txt")
    
    if not os.path.exists(route_dir):
        return 
    
    with open(meta_file, 'w') as file:
        file.write(metadata)

def read_route_meta(route_name):
    route_dir = os.path.join(dataset_root, route_name)
    
    if not os.path.exists(route_dir):
        return ROUTE_META

    meta_file = os.path.join(route_dir, "metadata.txt")
    try:
        with open(meta_file, 'r') as file:
            return file.read()
    except:
        return ROUTE_META
    
def write_video_meta(metadata, route_name, video_name):
    route_dir = os.path.join(dataset_root, route_name)
    meta_file = os.path.join(route_dir, "raw_video", os.path.splitext(video_name)[0] + ".txt")
    video_file = os.path.join(route_dir, "raw_video", video_name)
    
    if not os.path.exists(video_file):
        return

    with open(meta_file, 'w') as file:
        file.write(metadata)
    
def read_video_meta(route_name, video_name):
    route_dir = os.path.join(dataset_root, route_name)
    meta_file = os.path.join(route_dir, "raw_video", os.path.splitext(video_name)[0] + ".txt")

    if video_name == NO_CHOICE_MSG:
        return ""

    if not os.path.exists(route_dir):
        return VIDEO_META

    try:
        with open(meta_file, 'r') as file:
            return file.read()
    except:
        return VIDEO_META

def upload_videos(videos, route_name, route_info, video_info, progress=gr.Progress(track_tqdm=True)):
    ui_update = {"route_name": gr.update(), "video_file": videos, "progress": "", 
                 "video_info": video_info, "route_info": route_info}
    
    # Validate Input
    if not videos:
        ui_update["progress"] = "**ERROR: Please upload at least one video!**"
        return ui_update, "-"

    if not ft.is_valid_directory_name(route_name):
        ui_update[
            "progress"
        ] = "**ERROR: Invalid route name!** The route name must be '_' separated alphabets + digits"
        return ui_update, "-"

    if is_route_example(route_name):
        ui_update["progress"] = "**ERROR: Example Routes may not be modified!"
        return ui_update, "-"

    videos = [videos]
    route_dir = os.path.join(dataset_root, route_name)
    video_route = os.path.join(route_dir, "raw_video")

    new_route = False
    if not os.path.exists(route_dir):
        os.makedirs(video_route)
        new_route = True

    def move_video(video):
        if video is not None:
            video_name = os.path.basename(video)
            destination_path = os.path.join(video_route, video_name)
            shutil.move(video, destination_path)
            return destination_path
        return None

    def clean_transaction(error_msg):
        if new_route:
            try:
                ft.delete_directory(route_dir)
            except:
                error_msg += "\n**ERROR Failed to clean transactions**"
        else:
            for video in videos:
                video_name = os.path.basename(video)
                video_path = os.path.join(video_route, video_name)
                try:
                    if os.path.exists(video_path) and os.path.isfile(video_path):
                        os.remove(video_path)
                except:
                    error_msg += "\n**ERROR Failed to clean transactions**"

            ui_update["progress"] = error_msg
            ui_update["video_file"] = None
            return ui_update, "-"

    ### Move Video ###
    progress(0, desc="Moving Videos...")

    skipped = list()
    uploaded = list()
    for video in videos:
        video_name = os.path.basename(video)
        if not is_valid_video(video):
            skipped.append(video_name)
            continue

        try:
            move_video(video)
        except:
            return clean_transaction("**ERROR: Failed to upload Video!**")
        uploaded.append(video_name)

    write_route_meta(route_info, route_name)
    write_video_meta(video_info, route_name, os.path.basename(videos[0]))

    msg = ""
    if skipped:
        msg += "**Invalid file(s):** " + ", ".join(skipped)
        msg += "\n\n"
    if uploaded:
        msg += "**Successfully uploaded following video(s):** "
        msg += ", ".join(uploaded)


    ui_update["progress"] = msg
    ui_update["video_file"] = None
    ui_update["video_info"] = VIDEO_META
    return ui_update, "-"


def update_non_example_routes():
    return gr.update(choices=non_example_routes())


def update_upload_ui(ui_update):
    return ui_update["video_file"], update_non_example_routes(), ui_update["progress"], ui_update["video_info"]

ROUTE_META = """Route Location:
Last Update (MM/DD/YYYY):
Last Edit by:
Description:

"""

VIDEO_META = """Route:
Uploader:
Recorded Date (MM/DD/YYYY):
Time-of-day: 
Weather Condition:
Description:

"""

def upload_tab():
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                gr.Markdown("# Upload Videos")
            with gr.Row():
                instruction = gr.Markdown(WELCOME_UPLOAD, label="Instructions")

        with gr.Column(scale=6):
            with gr.Row():
                route_name_input = gr.Dropdown(
                    choices=non_example_routes(),
                    label="Route Name",
                    value="str",
                    allow_custom_value=True,
                    info="You can choose from the existing route or add a new route. \nMust be '_' separated alphabets + digits",
                )
            with gr.Row():
                route_info = gr.TextArea(
                    label="Enter Route Infomation",
                    value=ROUTE_META,
                    interactive=True
                    )
                video_info = gr.TextArea(
                    label="Enter Video Infomation",
                    value=VIDEO_META,
                    interactive=True
                )

            with gr.Row():
                video_file = gr.File(
                    label="Upload Video", file_count="single", type="filepath", file_types=["video"]
                )
            with gr.Row():
                register_btn = gr.Button("Upload")

    with gr.Row():
        progess_title = gr.Markdown("## Progress")
    with gr.Row():
        progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")

    ui_updates = gr.State()

    register_btn.click(
        upload_videos,
        inputs=[video_file, route_name_input, route_info, video_info],
        outputs=[ui_updates, progress],
    )
    route_name_input.change(
        update_non_example_routes,
        inputs=[],
        outputs=route_name_input,
        show_progress=False,
    )

    route_name_input.change(
        read_route_meta,
        inputs=route_name_input,
        outputs=route_info,
        show_progress=False
    )

    progress.change(
        update_upload_ui,
        inputs=ui_updates,
        outputs=[video_file, route_name_input, progress, video_info],
    )

WELCOME_FILE = """
1. Choose a specific route to view detailed information about it.
2. Select a video file to access and display its associated information.
3. You can edit information then press "Save" button to modify the information.
"""

def files_tab():
    with gr.Row():
        gr.Markdown("# Check out current dataset")
    with gr.Row():
        instruction = gr.Markdown(WELCOME_FILE, label="Instructions")

    with gr.Row():
        route_name_input = gr.Dropdown(
            choices=ft.get_directories(dataset_root),
            label="Choose Route",
            interactive=True,
            allow_custom_value=False,
            value="str",
        )
        video_name = gr.Dropdown(
            choices=[SELECT_ROUTE_MSG],
            label="Choose Videos",
            allow_custom_value=False,
            interactive=True,
            value="str",
        )
    with gr.Row():
        route_info = gr.TextArea(
            label="Route Infomation",
            value="Choose Route",
            interactive=True
            )
        video_info = gr.TextArea(
            label="Video information",
            value="Choose Video",
            interactive=True
        )
    with gr.Row():
        save_btn = gr.Button(
            "Save",
            interactive=True
        )

    save_btn.click(
        write_route_meta,
        inputs=[route_info, route_name_input],
        outputs=None
    )

    save_btn.click(
        write_video_meta,
        inputs=[video_info, route_name_input, video_name],
        outputs=None
    )

    route_name_input.change(
        get_video_files_for_route,
        inputs=route_name_input,
        outputs=[video_name],
        show_progress=False,
    )

    route_name_input.change(
        lambda x: gr.update(choices=ft.get_directories(dataset_root)),
        inputs = route_name_input,
        outputs= route_name_input,
        show_progress=False,
    )

    route_name_input.change(
        read_route_meta,
        inputs=route_name_input,
        outputs=route_info,
        show_progress=False
    )

    video_name.change(
        read_video_meta,
        inputs=[route_name_input, video_name],
        outputs=video_info
    )


WELCOME_RESULT = """# Explore Results
1. Select the Route, along with the Database and Query Videos.
2. (Optional) activate the "Overlay" feature to create aligned image pairs.
4. Navigate through frames using the "Playback Control".
5. Utilize the "Object Detection Control" to identify objects within the frames.
"""

def load_overlay(idx, inputs):
    if inputs == None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    
    matches, _, db_frame_list, aligned_query_frame_list = inputs
    
    query_idx_orig, database_index_aligned = matches[idx]

    query_img = cv2.imread(aligned_query_frame_list[query_idx_orig])
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    db_img = cv2.imread(db_frame_list[database_index_aligned])
    db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)

    blended = blend_img(0.5, query_img, db_img)
    return blended, blended, query_img, db_img

def display_images(idx, inputs, overlay_mode):
    matches, query_frame_list, db_frame_list, overlay_query_frame_list = inputs


    query_len = len(query_frame_list)
    db_len = len(db_frame_list)

    query_idx_orig, database_index_aligned = matches[idx]
    if overlay_mode:
        query_img_orig = overlay_query_frame_list[query_idx_orig]
    else:
        query_img_orig = query_frame_list[query_idx_orig]

    database_img_aligned = db_frame_list[database_index_aligned]

    q_total_time, show_hours = format_time(
        query_len / 2, show_hours=False, final_time=True
    )
    q_curr_time, _ = format_time(query_idx_orig / 2, show_hours)
    txt_query_idx = f"Frame Index: {query_idx_orig}/{query_len}"
    txt_query_time = f"Playback Time: {q_curr_time}/{q_total_time}"

    db_total_time, show_hours = format_time(db_len / 6, final_time=True)
    db_curr_time, _ = format_time(database_index_aligned / 6, show_hours)
    txt_db_idx = f"Frame Index: {database_index_aligned}/{db_len}"
    txt_db_time = f"Playback Time: {db_curr_time}/{db_total_time}"

    return (
        txt_query_idx,
        txt_query_time,
        txt_db_idx,
        txt_db_time,
        query_img_orig,
        database_img_aligned,
    )

def _run_alignment(route_name, database_video_name, query_video_name, overlay):
    database_video_name = glob(
        os.path.join(dataset_root, route_name, "raw_video", database_video_name + ".*")
    )
    if not database_video_name:
        return [], [], []
    else:
        database_video_name = [file for file in database_video_name if not file.endswith(".txt")]
        if not database_video_name:
            return [], [], []
        database_video_name = os.path.basename(database_video_name[0])

    query_video_name = glob(
        os.path.join(dataset_root, route_name, "raw_video", query_video_name + ".*")
    )
    if not query_video_name:
        return [], [], []
    else:
        query_video_name = [file for file in query_video_name if not file.endswith(".txt")]
        if not query_video_name:
            return [], [], []
        query_video_name = os.path.basename(query_video_name[0])

    ## Load DTW and VLAD Features ##
    database_video = DatabaseVideo(
        datasets_dir=dataset_root, route_name=route_name, video_name=database_video_name
    )
    query_video = QueryVideo(
        datasets_dir=dataset_root, route_name=route_name, video_name=query_video_name
    )

    db_frame_list = database_video.get_frames()
    query_frame_list = query_video.get_frames()
    overlay_query_frame_list = query_frame_list

    global grounding_sam
    if grounding_sam:
        del grounding_sam
        grounding_sam = None

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
    return (matches, query_frame_list, db_frame_list, overlay_query_frame_list)


def run_detection(idx, inputs, text_prompt, box_threshold, text_threshold):
    matches, query_frame_list, db_frame_list, _ = inputs

    query_idx_orig, database_index_aligned = matches[idx]

    query_img_orig = query_frame_list[query_idx_orig]
    database_img_aligned = db_frame_list[database_index_aligned]

    query_annotate = "./tmp/annotated_query.jpg"
    db_annotate = "./tmp/annotated_db.jpg"

    global grounding_sam
    if not grounding_sam:
        grounding_sam = GoundingSAMEngine(torch.device("cuda:0"))

    grounding_sam.generate_masked_images(
        query_img_orig, text_prompt, query_annotate, box_threshold, text_threshold
    )
    grounding_sam.generate_masked_images(
        database_img_aligned, text_prompt, db_annotate, box_threshold, text_threshold
    )

    return (query_annotate, db_annotate)



def run_alignment(route, query, db, overlay, progress=gr.Progress(track_tqdm=True)):
    msg = ""
    allow_overlay = gr.update(interactive=overlay, value=False)
    # Validate Input
    if not route or route == "str":
        msg = "**ERROR: Please Choose your route and videos!**"
        return ([], [], []), *update_ui_result(None), msg, allow_overlay
    elif query in [NO_CHOICE_MSG, SELECT_ROUTE_MSG, "str"]:
        msg = "**ERROR: Please Choose your query video!**"
        return ([], [], []), *update_ui_result(None), msg, allow_overlay
    elif db in [NO_CHOICE_MSG, SELECT_ROUTE_MSG, "str"]:
        msg = "**ERROR: Please Choose your db video!**"
        return ([], [], []), *update_ui_result(None), msg, allow_overlay

    result = _run_alignment(route, query, db, overlay)
    return result, *update_ui_result(result), "**Running**", allow_overlay


def update_ui_result(matches):
    if matches:
        idx = matches[0]
        query = matches[1]
        db = matches[2]
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            # gr.update(visible=True, maximum=len(idx), value=0),
            gr.Slider(
                0,
                len(idx),
                value=0,
                step=1,
                label="Choose Query Frame Index",
                visible=True,
                scale=8
            ),
            gr.update(visible=True),
            gr.update(visible=True),
            query[idx[0][0]],
            db[idx[0][1]],
        )
    else:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            None,
        )


def get_analyzed_video_for_route(route):
    if route and route != "str":
        return (
            get_analyzed_list(route, "database"),
            get_analyzed_list(route, "query"),
        )

    return (
        gr.update(choices=[SELECT_ROUTE_MSG]),
        gr.update(choices=[SELECT_ROUTE_MSG]),
    )


def get_analyzed_list(route, video_type: Literal["database", "query"]):
    target_dir = os.path.join(dataset_root, route, video_type)
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        return gr.update(choices=list(os.listdir(target_dir)))
    return gr.update(choices=[NO_CHOICE_MSG])

def blend_img(idx, query, db): 
    return (query * idx + db * (1-idx)).astype(np.uint8)

def result_tab(demo):
    matching_states = gr.State(None)
    run_state = gr.State(False)
    slider_idx = gr.Number(0, visible=False)
    query_image_overlay = gr.Image(type="numpy", visible=False)
    db_image_overlay = gr.Image(type="numpy", visible=False)
    blended_overlay = gr.Image(type="numpy", visible=False)


    with gr.Row():
        with gr.Column(scale=5):
            # with gr.Row():
            #     gr.Markdown("### Explore Results")
            # with gr.Row():
            gr.Markdown(WELCOME_RESULT, label="Instructions")
        with gr.Column(scale=5):
            with gr.Row():
                route_name_input = gr.Dropdown(
                    choices=ft.get_directories(dataset_root),
                    label="Route",
                    interactive=True,
                    allow_custom_value=False,
                    value="str",
                    scale=1
                )
                database_video_name = gr.Dropdown(
                    choices=[SELECT_ROUTE_MSG],
                    label="Database Video",
                    allow_custom_value=False,
                    interactive=True,
                    value="str",
                    scale=1
                )
                query_video_name = gr.Dropdown(
                    choices=[SELECT_ROUTE_MSG],
                    label="Query Video",
                    allow_custom_value=False,
                    interactive=True,
                    value="str",
                    scale=1
                )
                overlay = gr.Checkbox(
                    label="Overlay", info="Check to compute warped frames for Overlay View. The result can be viewed later in the 'Overlay View' or with 'Aligned Mode'.",
                    scale=3
                )
            with gr.Row():
                run_btn = gr.Button("Run")

    with gr.Row():
        query_img_orig = gr.Image(type="filepath", label="Query Frame", container=True)
        db_img_aligned = gr.Image(
            type="filepath", label="Database Match", container=True
        )
        
        padding1 = gr.Textbox("",container=False,visible=False,scale=1,interactive=False)
        overlay_img = gr.Image(type="numpy", label="Overlay View", visible=False, scale=3)
        padding2 = gr.Textbox("",container=False,visible=False,scale=1,interactive=False)

    with gr.Row():
        query_index = gr.Textbox("", container=False, scale=1)
        query_playback_time = gr.Textbox("", text_align="right", container=False, scale=1)
        db_index = gr.Textbox("", container=False)
        db_playback_time = gr.Textbox("", text_align="right", container=False, scale=1)
    
    def two_image_view():
        return {query_img_orig: gr.update(visible=True),
                db_img_aligned: gr.update(visible=True),
                overlay_img: gr.update(visible=False),
                padding1: gr.update(visible=False),
                padding2: gr.update(visible=False),
                query_index: gr.update(visible=True),
                query_playback_time: gr.update(visible=True),
                db_index: gr.update(visible=True),
                db_playback_time: gr.update(visible=True)
                }

    def one_image_view():
        return {query_img_orig: gr.update(visible=False),
                db_img_aligned: gr.update(visible=False),
                overlay_img: gr.update(visible=True),
                padding1: gr.update(visible=True),
                padding2: gr.update(visible=True),
                query_index: gr.update(visible=False),
                query_playback_time: gr.update(visible=False),
                db_index: gr.update(visible=False),
                db_playback_time: gr.update(visible=False)
                }
    
    



    with gr.Tab("Playback Control") as result_playback_tab:
        result_playback_tab.select(two_image_view, 
                                   inputs=None,
                                   outputs=[
                                       query_img_orig,
                                       db_img_aligned,
                                       overlay_img,
                                       padding1,
                                       padding2,
                                       query_index,
                                       query_playback_time,
                                       db_index,
                                       db_playback_time
                                   ], show_progress=False)

        with gr.Row():
            play_btn = gr.Button("Start", visible=False, scale=1, icon=os.path.join(icons_dir,"play.png"))
            slider = gr.Slider(
                0,
                1,
                value=1,
                step=1,
                label="Choose Query Frame Index",
                visible=False,
                scale=8
            )
            play_overlay_mode = gr.Checkbox(
                label="Aligned Mode", info="Check to display warped version of the query frames",
                interactive=False
            )
            
        with gr.Row():
            prev_5_btn = gr.Button("-5", visible=False, icon=os.path.join(icons_dir,"replay5.png"))
            prev_1_btn = gr.Button("-1", visible=False, icon=os.path.join(icons_dir,"replay.png"))
            next_1_btn = gr.Button("+1", visible=False, icon=os.path.join(icons_dir,"forward.png"))
            next_5_btn = gr.Button("+5", visible=False, icon=os.path.join(icons_dir,"forward5.png"))
    
        
    with gr.Tab("Object Detection Control") as result_detection_tab:
        result_detection_tab.select(two_image_view, 
                                   inputs=None,
                                   outputs=[
                                       query_img_orig,
                                       db_img_aligned,
                                       overlay_img,
                                       padding1,
                                       padding2,
                                       query_index,
                                       query_playback_time,
                                       db_index,
                                       db_playback_time
                                   ], show_progress=False)
        
        with gr.Row():
            text_prompt = gr.Textbox(
                label="Detection Prompt[Seperate unique classes with '.', e.g: cat . dog . chair]",
                placeholder="Cannot be empty",
                value="objects",
                scale=8
            )
            detect_button = gr.Button("Detect", visible=True, scale=2)
        
        with gr.Row():
            box_threshold = gr.Slider(
                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.30, step=0.01
                )
            text_threshold = gr.Slider(
                label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.01
            )

    with gr.Tab("Overlay View") as result_overlay_tab:
        result_overlay_tab.select(one_image_view, 
                                   inputs=None,
                                   outputs=[
                                       query_img_orig,
                                       db_img_aligned,
                                       overlay_img,
                                       padding1,
                                       padding2,
                                       query_index,
                                       query_playback_time,
                                       db_index,
                                       db_playback_time
                                   ], show_progress=True)
        with gr.Row():
            gr.Textbox("",container=False,visible=False,scale=1,interactive=False)
            overlay_0 = gr.Button("Query Frame", scale=1)
            overlay_half = gr.Button("Overlay Frame", scale=1)
            overlay_1 = gr.Button("Database Frame", scale=1)
            gr.Textbox("",container=False,visible=False,scale=1,interactive=False)

        result_overlay_tab.select(load_overlay,
                                  inputs=[slider, matching_states],
                                  outputs=[
                                      overlay_img,
                                      blended_overlay,
                                      query_image_overlay,
                                      db_image_overlay,
                                  ],
                                  show_progress=False
                                  )


    route_name_input.change(
        get_analyzed_video_for_route,
        inputs=route_name_input,
        outputs=[database_video_name, query_video_name],
        show_progress=False,
    )

    route_name_input.change(
        lambda x: gr.update(choices=ft.get_directories(dataset_root)),
        inputs = route_name_input,
        outputs= route_name_input,
        show_progress=False,
    )

    run_btn.click(
        run_alignment,
        inputs=[route_name_input, database_video_name, query_video_name, overlay],
        outputs=[
            matching_states,
            prev_1_btn,
            prev_5_btn,
            play_btn,
            slider,
            next_1_btn,
            next_5_btn,
            query_img_orig,
            db_img_aligned,
            query_index,
            play_overlay_mode,
        ],
        show_progress="full",
    )

    def reset_slider():
        slider.value = 0

    run_btn.click(
        reset_slider,
        show_progress=False
    )

    def prev_click(input, step):
        slider.value = slider.value - step
        if slider.value <= 0:
            slider.value = 0
        return slider.value

    def next_click(input, step):
        slider.value = slider.value + step
        if slider.value >= len(input[0]):
            slider.value = len(input[0]) - 1
        return slider.value

    click_1 = prev_1_btn.click(
        prev_click,
        inputs=[matching_states, gr.State(1)],
        outputs=[slider],
        trigger_mode="always_last",
        show_progress=False,
    )

    click_2 = prev_5_btn.click(
        prev_click,
        inputs=[matching_states, gr.State(5)],
        outputs=[slider],
        trigger_mode="always_last",
        show_progress=False,
    )

    click_3 = next_1_btn.click(
        next_click,
        inputs=[matching_states, gr.State(1)],
        outputs=[slider],
        trigger_mode="always_last",
        show_progress=False,
    )

    click_4 = next_5_btn.click(
        next_click,
        inputs=[matching_states, gr.State(5)],
        outputs=[slider],
        trigger_mode="always_last",
        show_progress=False,
    )


    detect_button.click(
        run_detection,
        inputs=[slider, matching_states, text_prompt, box_threshold, text_threshold],
        outputs=[query_img_orig, db_img_aligned],
    )

    def inc_local():
        if run_state.value:
            slider.value = slider.value + 1
            return slider.value
        return gr.update()
    
    def toggle_run(play_btn):
        if play_btn == "Start":
            run_state.value = True
            return gr.update(icon=os.path.join(icons_dir,"pause.png"), value="Stop"), gr.update(interactive=False), gr.update(interactive=False)
        else:
            run_state.value = False
            return gr.update(icon=os.path.join(icons_dir,"play.png"), value="Start"), gr.update(interactive=True), gr.update(interactive=True)

    def set_slide(idx):
        slider.value = idx

    demo.load(inc_local, inputs=None, outputs=slider_idx, every=0.5)
    slider.release(set_slide, inputs=slider, outputs=None, show_progress=False)
    slider_idx.change(lambda x:x, inputs=slider_idx, outputs=slider, show_progress=False)
    click_5 = play_btn.click(toggle_run, inputs=play_btn, outputs=[play_btn, slider, run_btn], show_progress=False)

    slider.change(
        display_images,
        inputs=[slider, matching_states, play_overlay_mode],
        outputs=[
            query_index,
            query_playback_time,
            db_index,
            db_playback_time,
            query_img_orig,
            db_img_aligned,
        ],
        show_progress=False,
        trigger_mode="multiple",
        cancels=[click_1, click_2, click_3, click_4, click_5]
    )

    overlay_0.click(lambda x:x, query_image_overlay, overlay_img, show_progress=False)
    overlay_half.click(lambda x:x, blended_overlay, overlay_img, show_progress=False)
    overlay_1.click(lambda x:x, db_image_overlay, overlay_img, show_progress=False)

    play_overlay_mode.select(
        display_images,
        inputs=[slider, matching_states, play_overlay_mode],
        outputs=[
            query_index,
            query_playback_time,
            db_index,
            db_playback_time,
            query_img_orig,
            db_img_aligned,
        ],
        show_progress=False,
        cancels=[click_1, click_2, click_3, click_4, click_5]
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="This program computes  ",
        epilog=dataset_layout_help,
    )
    # ... add arguments to parser ...
    parser.add_argument(
        "--dataset-root", default="../data", help="Root directory of datasets"
    )
    parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=True)
    parser.add_argument("--cuda_device", help="Select cuda device", default=0, type=int)
    args = parser.parse_args()

    global dataset_root
    dataset_root = args.dataset_root

    global torch_device
    available_gpus = torch.cuda.device_count()
    print(f"Avaliable GPU={available_gpus}")
    if args.cuda and available_gpus > 0:
        cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
        torch_device = torch.device("cuda:" + str(cuda_device))
    else:
        torch_device = torch.device("cpu")

    print(torch_device)

    with gr.Blocks() as demo:
        gr.Markdown("# Quetzal: Drone Footages Frame Alignment")
        with gr.Tab("File Explorer"):
            files_tab()
        with gr.Tab("Upload Videos"):
            upload_tab()
        with gr.Tab("Analyze Videos"):
            analyze_tab()
        with gr.Tab("Alignment Results"):
            result_tab(demo)
        with gr.Tab("Delete Files"):
            delete_tab()

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()

# pip install markupsafe==2.0.1


# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
