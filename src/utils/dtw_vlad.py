import faiss
import numpy as np
from copy import deepcopy

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
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
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
    for query_idx, db_idx in (zip(*path)):
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
            if abs(curr_mv_avg - avg_of_k) < (2.3/query_fps):
                new_time = query_idx/query_fps + curr_mv_avg 
                new_db_idx = int(new_time * db_fps + 0.5)

        smooth_lst.append((query_idx, new_db_idx))
        total_diff += abs(db_idx - new_db_idx)

    return smooth_lst, total_diff