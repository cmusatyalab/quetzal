import gradio as gr
from src.video import *
import logging
from copy import deepcopy
from src.engines.vpr_engine.anyloc_engine import generate_VLAD, AnyLocEngine
from src.utils.dtw_vlad import *
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
    database_folder = glob(
        os.path.join(delete_dir, "database", os.path.splitext(video)[0])
    )
    query_folder = glob(os.path.join(delete_dir, "query", os.path.splitext(video)[0]))

    if os.path.isfile(video_file):
        os.remove(video_file)

    if database_folder and os.path.isdir(database_folder[0]):
        ft.delete_directory(database_folder[0])

    if query_folder and os.path.isdir(query_folder[0]):
        ft.delete_directory(query_folder[0])

    if ft.is_directory_effectively_empty(delete_dir):
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
            return gr.update(choices=file_list), gr.update(choices=non_example_routes())

    return gr.update(choices=[NO_CHOICE_MSG]), gr.update(choices=non_example_routes())


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
        outputs=[video_name, route_name_input],
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

    generate_VLAD(db_video, query_video, torch_device)

    return "**Success!**"


def analyze_tab():
    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                gr.Markdown("# Analyze Videos")
            with gr.Row():
                instruction = gr.Markdown(WELCOME_ANALYZE, label="Instructions")
            with gr.Row():
                progess_title = gr.Markdown("## Progress")
            with gr.Row():
                progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")

        with gr.Column(scale=4):
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
                video_type_input = gr.Dropdown(
                    choices=["database", "query"],
                    label="Video Type",
                    value="str",
                    interactive=True,
                    allow_custom_value=False,
                )
            with gr.Row():
                analyze_btn = gr.Button("Analyze")

    route_name_input.change(
        get_video_files_for_route,
        inputs=route_name_input,
        outputs=[video_name, route_name_input],
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
   - You can upload a new video for an existing route, or you can add a new route entirely.
   - Note: Modification is restricted for the pre-defined 'example' routes*. 

2. **Choose Your Video File(s):**
   - You have the option to upload multiple source videos.
   - Do not touch the interface while the files are being uploaded.

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


def upload_videos(videos, route_name, progress=gr.Progress(track_tqdm=True)):
    ui_update = {"route_name": gr.update(), "video_file": videos, "progress": ""}

    # Validate Input
    if not videos:
        ui_update["progress"] = "**ERROR: Please upload at least one video!**"
        print(ui_update["progress"])
        return ui_update, "-"

    if not ft.is_valid_directory_name(route_name):
        ui_update[
            "progress"
        ] = "**ERROR: Invalid route name!** The route name must be '_' separated alphabets + digits"
        return ui_update, "-"

    if is_route_example(route_name):
        ui_update["progress"] = "**ERROR: Example Routes may not be modified!"
        return ui_update, "-"

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

    msg = ""
    if skipped:
        msg += "**Invalid file(s):** " + ", ".join(skipped)
        msg += "\n\n"
    if uploaded:
        msg += "**Successfully uploaded following video(s):** "
        msg += ", ".join(uploaded)

    ui_update["progress"] = msg
    ui_update["video_file"] = None
    return ui_update, "-"


def update_non_example_routes():
    return gr.update(choices=non_example_routes())


def update_upload_ui(ui_update):
    return ui_update["video_file"], update_non_example_routes(), ui_update["progress"]


def upload_tab():
    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                gr.Markdown("# Upload Videos")
            with gr.Row():
                instruction = gr.Markdown(WELCOME_UPLOAD, label="Instructions")
            with gr.Row():
                progess_title = gr.Markdown("## Progress")
            with gr.Row():
                progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")

        with gr.Column(scale=4):
            with gr.Row():
                route_name_input = gr.Dropdown(
                    choices=non_example_routes(),
                    label="Route Name",
                    value="str",
                    allow_custom_value=True,
                    info="You can choose from the existing route or add a new route. \nMust be '_' separated alphabets + digits",
                )
            # with gr.Row():
            #     db_video_upload = gr.File(label="Upload Database Video")
            # with gr.Row():
            #     query_video_upload = gr.File(label="Upload Query Video")
            with gr.Row():
                video_file = gr.File(
                    label="Upload Video", file_count="multiple", type="filepath"
                )
            with gr.Row():
                register_btn = gr.Button("Upload and Analyze")

    ui_updates = gr.State()

    register_btn.click(
        upload_videos,
        inputs=[video_file, route_name_input],
        outputs=[ui_updates, progress],
    )
    route_name_input.change(
        update_non_example_routes, inputs=[], outputs=route_name_input
    )
    progress.change(
        update_upload_ui,
        inputs=ui_updates,
        outputs=[video_file, route_name_input, progress],
    )


WELCOME_RESULT = """
1. Select the Route
2. Select Database and Query Videos
3. Press **Run**
4. Navigate using the slider and the Previous/Next buttons
"""


def display_images(idx, inputs):
    matches, query_frame_list, db_frame_list = inputs

    query_len = len(query_frame_list)
    db_len = len(db_frame_list)

    query_idx_orig, database_index_aligned = matches[idx]

    query_img_orig = query_frame_list[query_idx_orig]
    database_img_aligned = db_frame_list[database_index_aligned]

    q_total_time, show_hours = format_time(
        query_len / 2, show_hours=False, final_time=True
    )
    q_curr_time, _ = format_time(query_idx_orig / 2, show_hours)
    txt_query_idx = f"### ||| Frame Index: {query_idx_orig}/{query_len} |||"
    txt_query_time = f"### ||| Playback Time: {q_curr_time}/{q_total_time} |||"

    db_total_time, show_hours = format_time(db_len / 6, final_time=True)
    db_curr_time, _ = format_time(database_index_aligned / 6, show_hours)
    txt_db_idx = f"### ||| Frame Index: {database_index_aligned}/{db_len} |||"
    txt_db_time = f"### ||| Playback Time: {db_curr_time}/{db_total_time} |||"

    return (
        query_img_orig,
        database_img_aligned,
        txt_query_idx,
        txt_query_time,
        txt_db_idx,
        txt_db_time,
    )


def left_click(idx, inputs):
    if idx > 0:
        idx = idx - 1
    return *display_images(idx, inputs), idx


def right_click(idx, inputs):
    matches = inputs[0]
    if idx < len(matches) - 1:
        idx = idx + 1

    return *display_images(idx, inputs), idx


def run_dtw(route_name, database_video_name, query_video_name, progress):
    database_video_name = glob(
        os.path.join(dataset_root, route_name, "raw_video", database_video_name + ".*")
    )
    if not database_video_name:
        return [], [], []
    else:
        database_video_name = os.path.basename(database_video_name[0])

    query_video_name = glob(
        os.path.join(dataset_root, route_name, "raw_video", query_video_name + ".*")
    )
    if not query_video_name:
        return [], [], []
    else:
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

    matches = [
        (query_frame_list[i], db_frame_list[i])
        for i in tqdm(range(len(query_frame_list)))
    ]
    anylocEngine = AnyLocEngine(
        database_video=database_video, query_video=query_video, device=torch_device
    )

    db_vlad = anylocEngine.get_database_vlad()
    query_vlad = anylocEngine.get_query_vlad()

    # Normalize and prepare x and y for FAISS
    db_vlad = F.normalize(db_vlad)
    query_vlad = F.normalize(query_vlad)
    cuda = torch_device != torch.device("cpu")
    try:
        db_indexes = create_FAISS_indexes(db_vlad.numpy(), cuda=cuda)
    except:
        db_indexes = create_FAISS_indexes(db_vlad.numpy(), cuda=False)

    ## Run DTW Algorithm using VLAD features ##
    _, _, D1, path = dtw(query_vlad.numpy(), db_vlad, db_indexes)
    matches = extract_unique_dtw_pairs(path, D1)

    # Smooth the frame alignment Results
    query_fps = query_video.get_fps()
    db_fps = database_video.get_fps()

    diff = 1
    count = 0
    k = 3
    while diff and count < 100:
        time_diff = [
            database_video.get_frame_time(d) - query_video.get_frame_time(q)
            for q, d in matches
        ]
        mv_avg = np.convolve(time_diff, np.ones(k) / k, mode="same")
        mv_avg = {k[0]: v for k, v in zip(matches, mv_avg)}
        matches, diff = smooth_frame_intervals(matches, mv_avg, query_fps, db_fps)
        count += 1

    return (
        matches,
        query_frame_list,
        db_frame_list,
    )


def run_matching(route, query, db, progress=gr.Progress(track_tqdm=True)):
    msg = "**Running**"
    # Validate Input
    if not route or route == "str":
        msg = "**ERROR: Please Choose your route and videos!**"
        return ([], [], []), *update_ui_result(None), msg
    elif query in [NO_CHOICE_MSG, SELECT_ROUTE_MSG, "str"]:
        msg = "**ERROR: Please Choose your query video!**"
        return ([], [], []), *update_ui_result(None), msg
    elif db in [NO_CHOICE_MSG, SELECT_ROUTE_MSG, "str"]:
        msg = "**ERROR: Please Choose your db video!**"
        return ([], [], []), *update_ui_result(None), msg

    result = run_dtw(route, query, db, progress)
    return result, *update_ui_result(result), "**Running**"


def update_ui_result(matches):
    if matches:
        idx = matches[0]
        query = matches[1]
        db = matches[2]
        return (
            gr.update(visible=True),
            gr.update(visible=True, maximum=len(idx), value=0),
            gr.update(visible=True),
            query[idx[0][0]],
            db[idx[0][1]],
        )
    else:
        return gr.update(), gr.update(), gr.update(), None, None


def get_analyzed_video_for_route(route):
    all_routes = ft.get_directories(dataset_root)

    if route and route != "str":
        return (
            get_analyzed_list(route, "database"),
            get_analyzed_list(route, "query"),
            gr.update(choices=all_routes),
        )

    return (
        gr.update(choices=[SELECT_ROUTE_MSG]),
        gr.update(choices=[SELECT_ROUTE_MSG]),
        gr.update(choices=all_routes),
    )


def get_analyzed_list(route, video_type: Literal["database", "query"]):
    target_dir = os.path.join(dataset_root, route, video_type)
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        return gr.update(choices=list(os.listdir(target_dir)))
    return gr.update(choices=["No choice"])


def result_tab():
    matching_states = gr.State()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                gr.Markdown("# Explore Results")
            with gr.Row():
                gr.Markdown(WELCOME_RESULT, label="Instructions")
        with gr.Column(scale=6):
            with gr.Row():
                route_name_input = gr.Dropdown(
                    choices=ft.get_directories(dataset_root),
                    label="Route",
                    interactive=True,
                    allow_custom_value=False,
                    value="str",
                )
                database_video_name = gr.Dropdown(
                    choices=[SELECT_ROUTE_MSG],
                    label="Database Video",
                    allow_custom_value=False,
                    interactive=True,
                    value="str",
                )
                query_video_name = gr.Dropdown(
                    choices=[SELECT_ROUTE_MSG],
                    label="Query Video",
                    allow_custom_value=False,
                    interactive=True,
                    value="str",
                )
            with gr.Row():
                run_btn = gr.Button("Run")

    with gr.Row():
        query_img_orig = gr.Image(type="filepath", label="Query Image")
        db_img_aligned = gr.Image(type="filepath", label="DataBase_aligned Image")
    with gr.Row():
        prev_btn = gr.Button("Previous Frame", scale=1, visible=False)
        slider = gr.Slider(
            0,
            1,
            value=1,
            step=1,
            label="Choose Query Frame Index",
            scale=4,
            visible=False,
        )
        next_btn = gr.Button("Next Frame", scale=1, visible=False)
    with gr.Row():
        query_index = gr.Markdown("## Frame Index & Playback Time")
        query_playback_time = gr.Markdown("", rtl=True)
        db_index = gr.Markdown("")
        db_playback_time = gr.Markdown("", rtl=True)

    route_name_input.change(
        get_analyzed_video_for_route,
        inputs=route_name_input,
        outputs=[database_video_name, query_video_name, route_name_input],
    )
    run_btn.click(
        run_matching,
        inputs=[route_name_input, database_video_name, query_video_name],
        outputs=[
            matching_states,
            prev_btn,
            slider,
            next_btn,
            query_img_orig,
            db_img_aligned,
            query_index,
        ],
        show_progress="full",
    )
    slider.release(
        display_images,
        inputs=[slider, matching_states],
        outputs=[
            query_img_orig,
            db_img_aligned,
            query_index,
            query_playback_time,
            db_index,
            db_playback_time,
        ],
    )

    prev_btn.click(
        left_click,
        inputs=[slider, matching_states],
        outputs=[
            query_img_orig,
            db_img_aligned,
            query_index,
            query_playback_time,
            db_index,
            db_playback_time,
            slider,
        ],
    )
    next_btn.click(
        right_click,
        inputs=[slider, matching_states],
        outputs=[
            query_img_orig,
            db_img_aligned,
            query_index,
            query_playback_time,
            db_index,
            db_playback_time,
            slider,
        ],
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
    parser.add_argument(
        "--cuda", action="store_true", help="Enable cuda", default=False
    )
    parser.add_argument("--cuda_device", help="Select cuda device", default=0, type=int)
    args = parser.parse_args()

    global dataset_root
    dataset_root = args.dataset_root

    global torch_device
    available_gpus = torch.cuda.device_count()
    if args.cuda and available_gpus > 0:
        cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
        torch_device = torch.device("cuda:" + str(cuda_device))
    else:
        torch_device = torch.device("cpu")

    with gr.Blocks() as demo:
        gr.Markdown("# Quetzal: Drone Footages Frame Alignment")
        with gr.Tab("Upload Videos"):
            upload_tab()
        with gr.Tab("Analyze Videos"):
            analyze_tab()
        with gr.Tab("Alignment Results"):
            result_tab()
        with gr.Tab("Delete Files"):
            delete_tab()

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()

# pip install markupsafe==2.0.1


# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
