import gradio as gr
from src.video import *
import logging
from copy import deepcopy 
from src.engines.vpr_engine.anyloc_engine import AnyLocEngine
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import src.utils.file_tools as ft
from glob import glob
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger("Main Process")
logger.setLevel(logging.DEBUG)

PASSWORD_FILE = 'password.hash'

dataset_root = "../data"
# torch_device = torch.device("cpu")
torch_device = torch.device("cuda:0")

# route_name = "army_demo"
# database_video_name = "P0370037.MP4" 
# query_video_name =  "P0400040.MP4" 

route_name = "example_mil19"
database_video_name = "P0190019.MP4" 
query_video_name =  "P0410041.MP4" #"P0200020.MP4" 

# route_name = "hot_metal_bridge"
# database_video_name = "P4330479.MP4" 
# query_video_name =  "P3406874.MP4" 

# route_name = "purdue"
# database_video_name = "Clip_26.mov" 
# query_video_name =  "Clip_27.mov" 

# route_name = "purdue_2"
# database_video_name = "Clip_24.mov" 
# query_video_name =  "Clip_25.mov" 
verbose = True

dataset_layout_help = '''
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
    '''

## Initialize System
def create_FAISS_indexes(db_vlad, chunk_size=1024):
    indexes = []
    for i in range(0, len(db_vlad), chunk_size):
        chunk = db_vlad[i:i + chunk_size]
        index = faiss.IndexFlatIP(chunk.shape[1])
        res = faiss.StandardGpuResources()  # Use GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(chunk)
        indexes.append(gpu_index)
    return indexes

def query_all_indexes(q_vec, indexes):
    q_vec = q_vec.reshape(1, -1)
    similarities = []
    for index in indexes:
        sim, indices = index.search(q_vec, index.ntotal)  # Query the full index
        ordered_similarities = np.zeros(index.ntotal)
        for sim, idx in zip(sim[0], indices[0]):
            ordered_similarities[idx] = sim
        similarities.append(ordered_similarities)
    combined_distances = 1 - np.concatenate(similarities, axis=0)
    return combined_distances

#addaped from # https://github.com/pierre-rouanet/dtw
def dtw(query_vlad, db_vlad, db_indexes, warp=1):
    assert len(query_vlad)
    assert len(db_vlad)

    r, c = len(query_vlad), len(db_vlad)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # View of D0 excluding the first row and column
    
    for i, q_vec in enumerate(query_vlad):
        distances = query_all_indexes(q_vec, db_indexes)
        D0[i + 1, 1:] = distances

    D1_orig = deepcopy(D1)
    # DTW computation
    for i in tqdm(range(r)):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j], D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)

    if len(query_vlad) == 1:
        path = np.zeros(len(db_vlad)), np.arange(len(db_vlad))
    elif len(db_vlad) == 1:
        path = np.arange(len(query_vlad)), np.zeros(len(query_vlad))
    else:
        path = _traceback(D0)

    return D1[-1, -1], D0[1:, 1:], D1_orig, path

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def extract_unique_dtw_pairs(path, cost=None):
    matches =   [(-1,-1)]
    for query_idx, db_idx in tqdm((zip(*path))):
        duplicates = []
        
        if matches[-1][0] != query_idx:
            if not duplicates: # empty duplicates
                matches.append((query_idx, db_idx))
            else:
                query_idx = matches[-1][0]
                
                if cost:
                    best_idx = min(duplicates, key = lambda x: cost[query_idx][x])
                    matches[-1][1] = duplicates[best_idx]
                else:
                    matches[-1][1] = duplicates[len(duplicates)//2]
                
                matches.append((query_idx, db_idx))
                duplicates = []
        else:
            duplicates.append(db_idx)
    return  matches[1:]

# Smoothing using frame-time-diff between the query and db video
# lst should congtain list of tuples, each tuple with (query frame index, db frame index)
# query frame idx must be sorted
def smooth_frame_intervals(lst, mv_avg_diff, query_fps, db_fps, k = 5):
    total_diff = 0
    smooth_lst = list()
    N = len(lst)
    for i, (query_idx, db_idx) in enumerate(lst):
        new_db_idx = db_idx
        curr_mv_avg = mv_avg_diff[query_idx] 

        # Get next k mv_avg
        next_k_mv_avg = list()
        for ii in range(i+1, i+k+1, 1):
            if ii < N:
                next_k_mv_avg.append(mv_avg_diff[lst[ii][0]])

        if len(next_k_mv_avg) != 0: 
            avg_of_k = sum(next_k_mv_avg)/len(next_k_mv_avg)

            # only adjust the index when change in mv_avg is consistent
            if abs(curr_mv_avg - avg_of_k) < (2.2/query_fps):
                new_time = query_idx/query_fps + curr_mv_avg 
                new_db_idx = int(new_time * db_fps + 0.5)

        smooth_lst.append((query_idx, new_db_idx))
        total_diff += abs(db_idx - new_db_idx)

    return smooth_lst, total_diff

def run_dtw(route_name, database_video_name, query_video_name):
    
    database_video_name = glob(os.path.join(dataset_root, route_name, "raw_video", database_video_name+".*"))
    if not database_video_name:
        return [], [], [], [], []
    else:
        database_video_name = os.path.basename(database_video_name[0])
        
    query_video_name = glob(os.path.join(dataset_root, route_name, "raw_video", query_video_name+".*"))
    if not query_video_name:
        return [], [], [], [], []
    else:
        query_video_name = os.path.basename(query_video_name[0])
    
    
    logger.info("Loading Videos")
    database_video = DatabaseVideo(datasets_dir=dataset_root,
                                    route_name=route_name,
                                    video_name=database_video_name)
    query_video = QueryVideo(datasets_dir=dataset_root,
                                route_name=route_name,
                                video_name=query_video_name)

    db_frame_list = database_video.get_frames()
    query_frame_list = query_video.get_frames()

    matches = [(query_frame_list[i], db_frame_list[i]) for i in range(len(query_frame_list))]
    anylocEngine = AnyLocEngine(database_video=database_video, query_video=query_video, device=torch_device)
    
    # dtw method
    db_vlad = anylocEngine.get_database_vlad()
    query_vlad = anylocEngine.get_query_vlad()

    # Normalize and prepare x and y for FAISS
    db_vlad = F.normalize(db_vlad)
    query_vlad = F.normalize(query_vlad)
    db_indexes = create_FAISS_indexes(db_vlad.numpy())
    _, _, D1, path = dtw(query_vlad.numpy(), db_vlad, db_indexes)
    matches = extract_unique_dtw_pairs(path, D1)

    query_fps = query_video.get_fps()
    db_fps = database_video.get_fps()

    
    diff = 1
    count = 0
    k = 3
    while diff and count < 100:
        time_diff = [database_video.get_frame_time(d) - query_video.get_frame_time(q) for q, d in matches]
        mv_avg = np.convolve(time_diff, np.ones(k) / k, mode="same")
        mv_avg = {k[0]:v for k, v in zip(matches, mv_avg)}
        matches, diff = smooth_frame_intervals(matches, mv_avg, query_fps, db_fps)
        count+=1
        
    return matches, query_frame_list, db_frame_list



def generate_VLAD(database_video: Video, query_video: Video):
    logger.info("Loading Videos")
    anylocEngine = AnyLocEngine(database_video=database_video, query_video=query_video, device=torch_device, mode='lazy')
    
    db_vlad = anylocEngine.get_database_vlad()
    query_vlad = anylocEngine.get_query_vlad()
    del anylocEngine
    
    return db_vlad, query_vlad
    
def is_route_example(route_name):
    if route_name in example_routes:
        return True
    return False
    
def delete_videos(route, video, confirm: str):
    if confirm != "confirm deletion":
        choice = [file for file in ft.get_directories(dataset_root) if not is_route_example(file)]
        return gr.update(choices=choice), gr.update(value=""), "**Error: The confirmation string does not match**"
    
    msg = ""

    delete_dir = os.path.join(dataset_root, route)
    video_file = os.path.join(delete_dir, "raw_video", video)
    database_folder = glob(os.path.join(delete_dir, "database", os.path.splitext(video)[0]))
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
    choice = [file for file in ft.get_directories(dataset_root) if not is_route_example(file)]
    return gr.update(choices=choice), gr.update(value=""), msg

def register_videos(query_video, db_video, route_name, progress=gr.Progress(track_tqdm=True)):

    if not query_video and not db_video:
        return (None, None, gr.update(), "**ERROR: Please upload at least one video!**"), "change"
    
    if not ft.is_valid_directory_name(route_name):
        return (query_video, db_video, gr.update(), "**ERROR: Invalid route name!** The route name must be '_' separated alphabets + digits"), "change"
    
    if is_route_example(route_name):
        return (query_video, db_video, gr.update(), "**ERROR: Example Routes may not be modified!"), "change"
        
    ### Move Video ###
    progress(0, desc="Moving Videos...")
    route_dir = os.path.join(dataset_root, route_name)
    video_route = os.path.join(route_dir, 'raw_video')
    
    new_route = False
    # Check if route already exists
    # if os.path.exists(route_dir):
    #     # Verify password
    #     password_file = os.path.join(dataset_root, route_name, PASSWORD_FILE)
    #     if not os.path.exists(password_file):
    #         return query_video, db_video, gr.update(), "**ERROR: Password file missing!**"

    #     with open(password_file, 'rb') as file:
    #         hashed_password = file.read()
    #         if not ft.check_password(hashed_password, password):
    #             return query_video, db_video, gr.update(), "**ERROR: Incorrect password!**"
    # else:
    #     # Create new route and hash password
    #     new_route = True
    #     os.makedirs(video_route)
    #     hashed_password = ft.hash_password(password)
    #     password_file = os.path.join(dataset_root, route_name, PASSWORD_FILE)
    #     with open(password_file, 'wb') as file:
    #         file.write(hashed_password)
    
    if not os.path.exists(route_dir):
        os.makedirs(video_route)
        new_route = True
            
    def move_video(video):
        if video is not None:
            original_path = video.name
            video_name = os.path.basename(original_path)
            destination_path = os.path.join(video_route, video_name)
            shutil.move(original_path, destination_path)
            return destination_path
        return None

    # Move Video files into data folder
    def clean_transaction(error_msg):
        if new_route:
            try: ft.delete_directory(route_dir)
            except: error_msg += "\n**ERROR Failed to clean transactions**"
        else:
            try: 
                if query_video_file_path:
                    video_name = os.path.basename(query_video_file_path)
                    if os.path.exists(query_video_file_path): 
                        os.remove(query_video_file_path)
                        frames_path = join(dataset_root, route_name, "query", os.path.splitext(video_name)[0])
                        if os.path.exists(frames_path): ft.delete_directory(frames_path)
                
                if db_video_file_path:
                    video_name = os.path.basename(db_video_file_path)
                    if os.path.exists(db_video_file_path): 
                        os.remove(db_video_file_path)
                        frames_path = join(dataset_root, route_name, "database", os.path.splitext(video_name)[0])
                        if os.path.exists(frames_path): ft.delete_directory(frames_path)
                
            except: error_msg += "\n**ERROR Failed to clean transactions**"
                
        return (None, None, gr.update(), error_msg), "change"
    
    query_video_file_path = None
    try: query_video_file_path = move_video(query_video)
    except: return clean_transaction("**ERROR: Failed to upload Query Video!**")

    db_video_file_path = None
    try: db_video_file_path = move_video(db_video)
    except: return clean_transaction("**ERROR: Failed to upload Database Video!**")

    progress(0, desc="Analyzing Video...")
    ## Generate VLAD ##
    if query_video: 
        try: query_video = QueryVideo(dataset_root, route_name, os.path.basename(query_video_file_path))
        except : return clean_transaction("**ERROR: Query Video file is not valid!**")
        
    if db_video: 
        try: db_video = DatabaseVideo(dataset_root, route_name, os.path.basename(db_video_file_path))
        except : return clean_transaction("**ERROR: Database Video file is not valid!**")
        
    try: generate_VLAD(database_video=db_video, query_video=query_video)
    except: return clean_transaction("**ERROR Failed to Generate VLAD**")

    routes = ft.get_directories(dataset_root)
    return (None, None, gr.update(choices=routes), "**Success!**"), "change"


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

example_routes = ["example_mil19", "example_army_demo", "example_purdue", "example_hot_metal_bridge"]
footage_domains = ["aerial", "indoor", "urban"] 

WELCOME_DELETE = '''
**Please follow the steps below:**

1. **Selecting Files to Delete:** 
   - Choose the route and its video files that you want to delete.
   - Note: Deletion is restricted for videos within 'example' routes (i.e., paths starting with 'example').

2. **Confirm Your Action**
   - To confirm deletion, enter the required phrase in the confirmation textbox.

3. **Initiate Deletion:**
   - Once confirmed, click the "Delete" button to permanently remove the selected files.
   
'''
def get_all_videos(route):
    if route:
        target_dir = os.path.join(dataset_root, route, "raw_video")
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            file_list = [os.path.basename(file) for file in glob(os.path.join(target_dir, "*"))]
            choice = [file for file in ft.get_directories(dataset_root) if not is_route_example(file)]
            return gr.update(choices=file_list), gr.update(choices=choice)
    
    choice = [file for file in ft.get_directories(dataset_root) if not is_route_example(file)]
    return gr.update(choices=["No choice"]), gr.update(choices=choice)

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
                route_name_input = gr.Dropdown(choices=[file for file in ft.get_directories(dataset_root) if not is_route_example(file)], 
                            label="Route", 
                            interactive=True,
                            allow_custom_value=False,
                            value="str")
                video_name = gr.Dropdown(choices=["select route"], 
                                label="Videos",
                                allow_custom_value=False, 
                                interactive=True,
                                value="str")
            with gr.Row():
                delete_confirm = gr.Textbox(interactive=True, label="Type 'confirm deletion', case sensitive")
            with gr.Row():
                delete_btn = gr.Button("Delete")
                
    route_name_input.change(get_all_videos, inputs=route_name_input, outputs=[video_name, route_name_input])

            # with gr.Row():
                # file_explorer = gr.FileExplorer(root=dataset_root, file_count='multiple', interactive=True, label="List of Routes", render=False)
    delete_btn.click(delete_videos, inputs=[route_name_input, video_name, delete_confirm], outputs=[route_name_input, delete_confirm, progress])

    # show_btn.click(lambda args: gr.update(visible=True, render=True), inputs=delete_confirm, outputs=file_explorer)
    # hide_btn.click(lambda args: gr.update(render=False, visible=False), inputs=delete_confirm, outputs=file_explorer)
    
WELCOME_UPLOAD = '''
**Welcome!** To begin, please follow the steps below:

1. **Select the Route:** 
   - You can upload a new video for an existing route, or you can add a new route entirely.
   - Note: Modification is restricted for the 'example' routes (i.e., paths starting with 'example').

2. **Choose Your Video File(s):**
   - You have the option to upload both database and query videos simultaneously.
   - You can also upload just one video and leave the other.

3. **Upload and Analyze:**
   - Click on the **"Upload and Analyze"** button to upload your dataset and proceed the analysis.
   - Do not touch the interface while the analysis is running.

Go to **"Alignment Results"** tab to see alignment results or **"Delete Files"** tab to delete your files!
Example Matches are prepared in the Alignment Results!
'''

def update_ui_upload(results):
    return results

def upload_tab():
    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                gr.Markdown("# Upload and Analyze Videos")
            with gr.Row():
                instruction = gr.Markdown(WELCOME_UPLOAD, label="Instructions")
            with gr.Row():
                progess_title = gr.Markdown("## Progress")
            with gr.Row():
                progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")
                
        with gr.Column(scale=4):
            with gr.Row():
                route_name_input = gr.Dropdown(choices=[file for file in ft.get_directories(dataset_root) if not is_route_example(file)], 
                            label="Route Name", 
                            value="str",
                            allow_custom_value=True,
                            info="You can choose from the existing route or add a new route. \nMust be '_' separated alphabets + digits")
            with gr.Row():
                db_video_upload = gr.File(label="Upload Database Video")
            with gr.Row():
                query_video_upload = gr.File(label="Upload Query Video")
            with gr.Row():
                register_btn = gr.Button("Upload and Analyze")
    
    results = gr.State()
            
    register_btn.click(register_videos, inputs=[query_video_upload, db_video_upload, route_name_input], outputs=[results, progress])
    route_name_input.change(lambda x: gr.update(choices=[file for file in ft.get_directories(dataset_root) if not is_route_example(file)]), inputs=route_name_input, outputs=route_name_input)
    progress.change(update_ui_upload, inputs=results, outputs=[query_video_upload, db_video_upload, route_name_input, progress])


WELCOME_RESULT = '''
1. **Select the Route** 
2. **Select Database and Query Videos**
3. **Press Run**
4. **Navigate using the slider and the Previous/Next buttons**
'''

def display_images(idx, inputs):
    matches, query_frame_list, db_frame_list = inputs
    
    query_len = len(query_frame_list)
    db_len = len(db_frame_list)
    
    query_idx_orig, database_index_aligned = matches[idx]
    
    query_img_orig = query_frame_list[query_idx_orig]
    database_img_aligned = db_frame_list[database_index_aligned]

    q_total_time, show_hours = format_time(query_len / 2, show_hours=False, final_time=True)
    q_curr_time, _= format_time(query_idx_orig / 2, show_hours)
    txt_query_idx = f"### ||| Frame Index: {query_idx_orig}/{query_len} |||"
    txt_query_time = f"### ||| Playback Time: {q_curr_time}/{q_total_time} |||"

    db_total_time, show_hours = format_time(db_len / 6, final_time=True)
    db_curr_time, _ = format_time(database_index_aligned / 6, show_hours)
    txt_db_idx = f"### ||| Frame Index: {database_index_aligned}/{db_len} |||"
    txt_db_time = f"### ||| Playback Time: {db_curr_time}/{db_total_time} |||"

    return query_img_orig, database_img_aligned, txt_query_idx, txt_query_time, txt_db_idx, txt_db_time

def left_click(idx, inputs):
    if idx > 0:
        idx = idx - 1
    return *display_images(idx, inputs), idx

def right_click(idx, inputs):
    matches = inputs[0]
    if idx < len(matches) - 1:
        idx = idx + 1
    
    return *display_images(idx, inputs), idx

def run_matching(route, query, db, progress=gr.Progress(track_tqdm=True)):
    result = run_dtw(route, query, db)
    return result, *update_ui_result(len(result[0])), "**Running**"

def update_ui_result(matches_len):
    return gr.update(visible=True), gr.update(visible=True, maximum=matches_len, value=0), gr.update(visible=True)

def get_video_lists(route):
    choice = [file for file in ft.get_directories(dataset_root)]
    return get_video_list(route, "database"), get_video_list(route, "query"), gr.update(choices=choice)

def get_video_list(route, video_type: Literal["database", "query"]):
    target_dir = os.path.join(dataset_root, route, video_type)
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        return gr.update(choices=list(os.listdir(target_dir)))
    return gr.update(choices=["No choice"])
        
def result_tab():
    matches = gr.State()
    query_frames = gr.State()
    db_frames = gr.State()
    query_video = gr.State()
    db_video = gr.State()
    
    matching_states = gr.State()
                
    with gr.Row():
        with gr.Column(scale = 4):
            with gr.Row():  
                gr.Markdown("# Explore Results")
            with gr.Row():
                gr.Markdown(WELCOME_RESULT, label="Instructions")
        with gr.Column(scale = 6):      
            with gr.Row():
                route_name_input = gr.Dropdown(choices=ft.get_directories(dataset_root), 
                            label="Route", 
                            interactive=True,
                            allow_custom_value=False,
                            value="str")
                database_video_name = gr.Dropdown(choices=["select route"], 
                            label="Database Video",
                            allow_custom_value=False, 
                            interactive=True,
                            value="str")
                query_video_name = gr.Dropdown(choices=["select route"], 
                            label="Query Video", 
                            allow_custom_value=False,
                            interactive=True,
                            value="str")
            with gr.Row():
                run_btn = gr.Button("Run")
                   
    with gr.Row():
        query_img_orig = gr.Image(type="filepath", label="Query Image")
        db_img_aligned = gr.Image(type="filepath", label="DataBase_aligned Image")
    with gr.Row():
            query_index = gr.Markdown("### Frame Index & Playback Time")
            query_playback_time = gr.Markdown("", rtl=True)
            db_index = gr.Markdown("")
            db_playback_time = gr.Markdown("", rtl=True)
    with gr.Row():
        prev_btn = gr.Button("Previous Frame", scale=1, visible=False)
        slider = gr.Slider(0, 1, step=1, label="Choose Query Frame Index", scale=4, visible=False)
        next_btn = gr.Button("Next Frame", scale=1, visible=False)

    route_name_input.change(get_video_lists, inputs=route_name_input, outputs=[database_video_name, query_video_name, route_name_input])
    run_btn.click(run_matching, 
                  inputs=[route_name_input, database_video_name, query_video_name], 
                  outputs=[matching_states, prev_btn, slider, next_btn, query_index], 
                  show_progress="full"
                  )
    slider.attach_load_event
    slider.release(display_images, 
                   inputs=[slider, matching_states],
                   outputs=[query_img_orig, db_img_aligned, query_index, query_playback_time, db_index, db_playback_time]
                   )

    prev_btn.click(left_click, inputs=[slider, matching_states], outputs=[query_img_orig, db_img_aligned, query_index, query_playback_time, db_index, db_playback_time, slider])
    next_btn.click(right_click, inputs=[slider, matching_states], outputs=[query_img_orig, db_img_aligned, query_index, query_playback_time, db_index, db_playback_time, slider])
    

with gr.Blocks() as demo:
    gr.Markdown("# Quetzal: Drone Footages Frame Alignment")
    with gr.Tab("Register Videos"):
        upload_tab()
    with gr.Tab("Alignment Results"):
        result_tab()
    with gr.Tab("Delete Files"):
        
        delete_tab()

    # upload_button.upload(upload_file, upload_button, file_output)
# pip install markupsafe==2.0.1
print(gr.__version__)
demo.queue()
demo.launch()
# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
