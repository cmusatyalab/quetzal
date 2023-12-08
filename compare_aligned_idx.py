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

logging.basicConfig()
logger = logging.getLogger("Main Process")
logger.setLevel(logging.DEBUG)

PASSWORD_FILE = 'password.hash'

dataset_root = "../data"
torch_device = torch.device("cpu")
# torch_device = torch.device("cuda:0")


# route_name = "army_demo"
# database_video_name = "P0370037.MP4" 
# query_video_name =  "P0400040.MP4" 

route_name = "mil19_orig"
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

# Load Video frames
# logger.info("Loading Videos")
# database_video = DatabaseVideo(datasets_dir=dataset_root,
#                                 route_name=route_name,
#                                 video_name=database_video_name)
# query_video = QueryVideo(datasets_dir=dataset_root,
#                             route_name=route_name,
#                             video_name=query_video_name)

# db_frame_list = database_video.get_frames()
# query_frame_list = query_video.get_frames()
# query_len = query_video.get_frame_idx(query_frame_list[-1])
# db_len = database_video.get_frame_idx(db_frame_list[-1])

# matches = [(query_frame_list[i], db_frame_list[i]) for i in range(len(query_frame_list))]
matches = [(1,1)]
# anylocEngine = AnyLocEngine(database_video=database_video, query_video=query_video, device=torch_device)

# # dtw method
# db_vlad = anylocEngine.get_database_vlad()
# query_vlad = anylocEngine.get_query_vlad()

# def create_FAISS_indexes(db_vlad, chunk_size=1024):
#     indexes = []
#     for i in range(0, len(db_vlad), chunk_size):
#         chunk = db_vlad[i:i + chunk_size]
#         index = faiss.IndexFlatIP(chunk.shape[1])
#         res = faiss.StandardGpuResources()  # Use GPU
#         gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
#         gpu_index.add(chunk)
#         indexes.append(gpu_index)
#     return indexes

# def query_all_indexes(q_vec, indexes):
#     q_vec = q_vec.reshape(1, -1)
#     similarities = []
#     for index in indexes:
#         sim, indices = index.search(q_vec, index.ntotal)  # Query the full index
#         ordered_similarities = np.zeros(index.ntotal)
#         for sim, idx in zip(sim[0], indices[0]):
#             ordered_similarities[idx] = sim
#         similarities.append(ordered_similarities)
#     combined_distances = 1 - np.concatenate(similarities, axis=0)
#     return combined_distances

# #addaped from # https://github.com/pierre-rouanet/dtw
# def dtw(query_vlad, db_vlad, db_indexes, warp=1):
#     assert len(query_vlad)
#     assert len(db_vlad)

#     r, c = len(query_vlad), len(db_vlad)
#     D0 = np.zeros((r + 1, c + 1))
#     D0[0, 1:] = np.inf
#     D0[1:, 0] = np.inf
#     D1 = D0[1:, 1:]  # View of D0 excluding the first row and column
    
#     for i, q_vec in enumerate(query_vlad):
#         distances = query_all_indexes(q_vec, db_indexes)
#         D0[i + 1, 1:] = distances

#     D1_orig = deepcopy(D1)
#     # DTW computation
#     for i in range(r):
#         for j in range(c):
#             min_list = [D0[i, j]]
#             for k in range(1, warp + 1):
#                 min_list += [D0[min(i + k, r), j], D0[i, min(j + k, c)]]
#             D1[i, j] += min(min_list)

#     if len(query_vlad) == 1:
#         path = np.zeros(len(db_vlad)), np.arange(len(db_vlad))
#     elif len(db_vlad) == 1:
#         path = np.arange(len(query_vlad)), np.zeros(len(query_vlad))
#     else:
#         path = _traceback(D0)

#     return D1[-1, -1], D0[1:, 1:], D1_orig, path

# def _traceback(D):
#     i, j = np.array(D.shape) - 2
#     p, q = [i], [j]
#     while (i > 0) or (j > 0):
#         tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
#         if tb == 0:
#             i -= 1
#             j -= 1
#         elif tb == 1:
#             i -= 1
#         else:  # (tb == 2):
#             j -= 1
#         p.insert(0, i)
#         q.insert(0, j)
#     return np.array(p), np.array(q)

# def extract_unique_dtw_pairs(path, cost=None):
#     matches =   [(-1,-1)]
#     for query_idx, db_idx in (zip(*path)):
#         duplicates = []
        
#         if matches[-1][0] != query_idx:
#             if not duplicates: # empty duplicates
#                 matches.append((query_idx, db_idx))
#             else:
#                 query_idx = matches[-1][0]
                
#                 if cost:
#                     best_idx = min(duplicates, key = lambda x: cost[query_idx][x])
#                     matches[-1][1] = duplicates[best_idx]
#                 else:
#                     matches[-1][1] = duplicates[len(duplicates)//2]
                
#                 matches.append((query_idx, db_idx))
#                 duplicates = []
#         else:
#             duplicates.append(db_idx)
#     return  matches[1:]

# # Normalize and prepare x and y for FAISS
# db_vlad = F.normalize(db_vlad)
# query_vlad = F.normalize(query_vlad)
# db_indexes = create_FAISS_indexes(db_vlad.numpy())
# _, _, D1, path = dtw(query_vlad.numpy(), db_vlad, db_indexes)
# matches = extract_unique_dtw_pairs(path, D1)

# # Smoothing using frame-time-diff between the query and db video
# # lst should congtain list of tuples, each tuple with (query frame index, db frame index)
# # query frame idx must be sorted
# def smooth_frame_intervals(lst, mv_avg_diff, query_fps, db_fps, k = 5):
#     total_diff = 0
#     smooth_lst = list()
#     N = len(lst)
#     for i, (query_idx, db_idx) in enumerate(lst):
#         new_db_idx = db_idx
#         curr_mv_avg = mv_avg_diff[query_idx] 

#         # Get next k mv_avg
#         next_k_mv_avg = list()
#         for ii in range(i+1, i+k+1, 1):
#             if ii < N:
#                 next_k_mv_avg.append(mv_avg_diff[lst[ii][0]])

#         if len(next_k_mv_avg) != 0: 
#             avg_of_k = sum(next_k_mv_avg)/len(next_k_mv_avg)

#             # only adjust the index when change in mv_avg is consistent
#             if abs(curr_mv_avg - avg_of_k) < (2.2/query_fps):
#                 new_time = query_idx/query_fps + curr_mv_avg 
#                 new_db_idx = int(new_time * db_fps + 0.5)

#         smooth_lst.append((query_idx, new_db_idx))
#         total_diff += abs(db_idx - new_db_idx)

#     return smooth_lst, total_diff

# query_fps = query_video.get_fps()
# db_fps = database_video.get_fps()

# diff = 1
# count = 0
# k = 3
# while diff and count < 100:
#     time_diff = [database_video.get_frame_time(d) - query_video.get_frame_time(q) for q, d in matches]
#     mv_avg = np.convolve(time_diff, np.ones(k) / k, mode="same")
#     mv_avg = {k[0]:v for k, v in zip(matches, mv_avg)}
#     matches, diff = smooth_frame_intervals(matches, mv_avg, query_fps, db_fps)
#     count+=1




def display_images(idx):
    query_idx_orig, database_index_aligned = matches[idx]

    query_img_orig = query_frame_list[query_idx_orig]
    database_img_aligned = db_frame_list[database_index_aligned]

    q_total_time, show_hours = format_time(query_video.get_frame_time(query_len), show_hours=False, final_time=True)
    q_curr_time, _= format_time(query_video.get_frame_time(query_idx_orig), show_hours)
    txt_query_idx = f"### ||| Frame Index: {query_idx_orig}/{query_len} |||"
    txt_query_time = f"### ||| Playback Time: {q_curr_time}/{q_total_time} |||"

    db_total_time, show_hours = format_time(database_video.get_frame_time(db_len), final_time=True)
    db_curr_time, _ = format_time(database_video.get_frame_time(database_index_aligned), show_hours)
    txt_db_idx = f"### ||| Frame Index: {database_index_aligned}/{db_len} |||"
    txt_db_time = f"### ||| Playback Time: {db_curr_time}/{db_total_time} |||"

    return query_img_orig, database_img_aligned, txt_query_idx, txt_query_time, txt_db_idx, txt_db_time


def left_click(idx):
    if idx > 0:
        idx = idx - 1
    return *display_images(idx), idx


def right_click(idx):
    if idx < len(matches) - 1:
        idx = idx + 1
    
    return *display_images(idx), idx


def generate_VLAD(database_video: Video, query_video: Video):
    logger.info("Loading Videos")
    anylocEngine = AnyLocEngine(database_video=database_video, query_video=query_video, device=torch_device, mode='lazy')
    
    if database_video: anylocEngine.get_database_vlad()
    if query_video: anylocEngine.get_query_vlad()
    
    del anylocEngine
    
def delete_videos(file_path, password):
    print(file_path.name)
    return None, file_path.name
    
def register_videos(query_video, db_video, route_name, password, progress=gr.Progress(track_tqdm=True)):
    if not query_video and not db_video:
        return None, None, gr.update(), "**ERROR: Please upload at least one video!**"
    
    if not ft.is_valid_directory_name(route_name):
        return query_video, db_video, gr.update(), "**ERROR: Invalid route name!** The route name must be '_' separated alphabets + digits"

    progress(0, desc="Moving Videos...")
    ### Move Video ###
    route_dir = os.path.join(dataset_root, route_name)
    video_route = os.path.join(route_dir, 'raw_video')
    
    new_route = False
    # Check if route already exists
    if os.path.exists(route_dir):
        # Verify password
        password_file = os.path.join(dataset_root, route_name, PASSWORD_FILE)
        if not os.path.exists(password_file):
            return query_video, db_video, gr.update(), "**ERROR: Password file missing!**"

        with open(password_file, 'rb') as file:
            hashed_password = file.read()
            if not ft.check_password(hashed_password, password):
                return query_video, db_video, gr.update(), "**ERROR: Incorrect password!**"
    else:
        # Create new route and hash password
        new_route = True
        os.makedirs(video_route)
        hashed_password = ft.hash_password(password)
        password_file = os.path.join(dataset_root, route_name, PASSWORD_FILE)
        with open(password_file, 'wb') as file:
            file.write(hashed_password)
            
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
                
        return None, None, gr.update(), error_msg
    
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
    return None, None, gr.update(choices=routes), "**Success!**"


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



example_routes = ["mil19", "army_demo", "purdue", "hot_metal_bridge"]
footage_domains = ["aerial", "indoor", "urban"] 

WELCOME_DELETE = '''
**Welcome!** To begin, please follow the steps below:

1. **Select the Route:** 
   - You can upload a new video for an existing route, or you can add a new route entirely.
   - 'mil19', 'army_demo', 'hot_metal_bridge', and 'purdue' routes are presented as examples, and you may not modify them.

2. **Choose Your Video File(s):**
   - You have the option to upload both database and query videos simultaneously, or upload just one video and leave the other.

3. **Remember Your Route Password:**
   - Each route is secured with a unique password. You'll need this password later to view results, add new videos, or delete the route.

4. **Upload and Analyze:**
   - After selecting your route, entering the password, and choosing your video(s), click on the **"Upload and Analyze"** button to proceed.
   - Also explore **"Alignment Results"** or **"Delete Files"** tabs 
'''

def delete_tab():
    with gr.Row():
        with gr.Column(scale=6):
            instruction = gr.Markdown(WELCOME_DELETE, label="Instructions")
        with gr.Column(scale=4):
            with gr.Row():
                progess_title = gr.Markdown("## Progress")
            with gr.Row():
                progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")
    with gr.Row():
        file_explorer = gr.FileExplorer(root=dataset_root, file_count='multiple', interactive=False, label="List of Routes")
        route_password = gr.Textbox("", label="Route Password", info="Stored in hash, but please do not use any personal password")
        delete_btn = gr.Button("Delete")
    
    delete_btn.click(delete_videos, inputs=[file_explorer, route_password], outputs=[file_explorer, progress])


WELCOME_UPLOAD = '''
**Welcome!** To begin, please follow the steps below:

1. **Select the Route:** 
   - You can upload a new video for an existing route, or you can add a new route entirely.
   - 'mil19', 'army_demo', 'hot_metal_bridge', and 'purdue' routes are presented as examples, and you may not modify them.

2. **Choose Your Video File(s):**
   - You have the option to upload both database and query videos simultaneously, or upload just one video and leave the other.

3. **Remember Your Route Password:**
   - Each route is secured with a unique password. This password is needed to view results, add new videos, or delete the route.

4. **Upload and Analyze:**
   - Click on the **"Upload and Analyze"** button to upload your dataset and proceed the analysis.

Go to **"Alignment Results"** tab to see alignment results or **"Delete Files"** tab to delete your files!
'''

def upload_tab(routes):
    with gr.Row():
        with gr.Column(scale=6):
            instruction = gr.Markdown(WELCOME_UPLOAD, label="Instructions")
        with gr.Column(scale=4):
            with gr.Row():
                progess_title = gr.Markdown("## Progress")
            with gr.Row():
                progress = gr.Markdown("\n\n\n\n\n\n\n", label="Progress")
    with gr.Row():

        with gr.Column(scale=1):
            
            route_name_input = gr.Dropdown(choices=routes, 
                                        label="Route Name", 
                                        value="str",
                                        allow_custom_value=True,
                                        info="You can choose from the existing route or add a new route. \nMust be '_' separated alphabets + digits")
            route_password = gr.Textbox("", label="Route Password", info="Stored in hash, but please do not use any personal password")
                        # route_domain_input = gr.Dropdown(choices=footage_domains, 
            #                         label="Footage Domain",
            #                         info="Choose the domain that best describes your drone footage")
        with gr.Column(scale=1):
            db_video_upload = gr.File(label="Upload Database Video")
        with gr.Column(scale=1):
            query_video_upload = gr.File(label="Upload Query Video")
    
            
    with gr.Row():
        register_btn = gr.Button("Upload and Analyze")
    
    register_btn.click(register_videos, inputs=[query_video_upload, db_video_upload, route_name_input, route_password], outputs=[query_video_upload, db_video_upload, route_name_input, progress])


def result_tab():
    with gr.Row():
        query_img_orig = gr.Image(type="filepath", label="Query Image")
        db_img_aligned = gr.Image(type="filepath", label="DataBase_aligned Image")
    with gr.Row():
            query_index = gr.Markdown("### Frame Index & Playback Time")
            query_playback_time = gr.Markdown("", rtl=True)
            db_index = gr.Markdown("")
            db_playback_time = gr.Markdown("", rtl=True)
    with gr.Row():
        prev_btn = gr.Button("Previous Frame", scale=1)
        slider = gr.Slider(0, len(matches), step=1, label="Choose Query Frame Index", scale=4)
        next_btn = gr.Button("Next Frame", scale=1)

    slider.release(display_images, inputs=slider, outputs=[query_img_orig, db_img_aligned, query_index, query_playback_time, db_index, db_playback_time])

    prev_btn.click(left_click, inputs=[slider], outputs=[query_img_orig, db_img_aligned, query_index, query_playback_time, db_index, db_playback_time, slider])
    next_btn.click(right_click, inputs=[slider], outputs=[query_img_orig, db_img_aligned, query_index, query_playback_time, db_index, db_playback_time, slider])


with gr.Blocks() as demo:
    routes = ft.get_directories(dataset_root)
    gr.Markdown("# Quetzal: Drone Footages Frame Alignment")
    with gr.Tab("Register Videos"):
        gr.Markdown("# Upload and Analyze Videos")
        upload_tab(routes)
    with gr.Tab("Alignment Results"):
        gr.Markdown("# Explore Results")
        result_tab()
    with gr.Tab("Delete Files"):
        gr.Markdown("# Delete Uploaded Files")
        delete_tab()

    # upload_button.upload(upload_file, upload_button, file_output)
# pip install markupsafe==2.0.1
print(gr.__version__)
demo.queue()
demo.launch()
# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
