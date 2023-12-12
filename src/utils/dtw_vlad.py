from copy import deepcopy
import numpy as np
from tqdm import tqdm
import faiss
from typing import List, Tuple, Optional


def create_FAISS_indexes(
    db_vlad: np.ndarray, chunk_size: int = 1024, cuda: bool = True
) -> List[faiss.IndexFlatIP]:
    """
    Creates FAISS indexes from the given VLAD vectors of a database, optionally using GPU acceleration.

    This function splits the database VLAD vectors into chunks and creates a FAISS index for each chunk.
    When CUDA is enabled, it utilizes GPU resources for the indexing process.

    Args:
    db_vlad (numpy.ndarray): The VLAD vectors for the database. Each row represents a VLAD vector.
    chunk_size (int, optional): The number of vectors in each chunk for the FAISS index. Default is 1024.
    cuda (bool, optional): Flag to enable or disable GPU acceleration. Default is True.

    Returns:
    list: A list of FAISS Index objects, each representing a chunk of the VLAD vectors.
    """
    indexes = []
    for i in range(0, len(db_vlad), chunk_size):
        chunk = db_vlad[i : i + chunk_size]
        index = faiss.IndexFlatIP(chunk.shape[1])
        if cuda:
            res = faiss.StandardGpuResources()  # Use GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(chunk)
        indexes.append(index)
    return indexes


def query_all_indexes(
    q_vec: np.ndarray, indexes: List[faiss.IndexFlatIP]
) -> np.ndarray:
    """
    Calculate distance between all FAISS indexes and a query vector and returns combined distances.

    Args:
    q_vec (numpy.ndarray): The query VLAD vector.
    indexes (list): A list of FAISS indexes to query.

    Returns:
    numpy.ndarray: Combined distances from all indexes.
    """
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


def dtw(
    query_vlad: np.ndarray,
    db_vlad: np.ndarray,
    db_indexes: List[faiss.IndexFlatIP],
    warp: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Performs Dynamic Time Warping (DTW) between query VLAD vectors and database VLAD vectors.
    modified from https://github.com/pierre-rouanet/dtw

    Args:
    query_vlad (numpy.ndarray): VLAD vectors for the query.
    db_vlad (numpy.ndarray): VLAD vectors for the database.
    db_indexes (list): FAISS indexes of the database VLAD vectors.
    warp (int, optional): The warping window width. Default is 1.

    Returns:
    tuple: A tuple containing the final DTW cost, the full DTW matrix, the original unwrapped DTW matrix, and the optimal path.
    """
    assert len(query_vlad)
    assert len(db_vlad)

    r, c = len(query_vlad), len(db_vlad)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # View of D0 excluding the first row and column

    for i in tqdm(range(r), desc="Calculating the distance matrix between the frames"):
        q_vec = query_vlad[i]
        distances = query_all_indexes(q_vec, db_indexes)
        D0[i + 1, 1:] = distances

    D1_orig = deepcopy(D1)
    # DTW computation
    for i in tqdm(range(r), desc="Aligning the frames"):
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


def _traceback(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Traces back the optimal path in the DTW matrix.

    Args:
    D (numpy.ndarray): The DTW matrix.

    Returns:
    tuple: A tuple of numpy arrays representing the indices of the optimal path through the matrix.
    """

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


def extract_unique_dtw_pairs(
    path: Tuple[np.ndarray, np.ndarray], cost: Optional[np.ndarray] = None
) -> List[Tuple[int, int]]:
    """
    Extracts unique pairs from the DTW path.

    Args:
    path (tuple): The DTW path as a tuple of numpy arrays.
    cost (numpy.ndarray, optional): The DTW cost matrix. Default is None.

    Returns:
    list: A list of unique pairs (tuples) from the DTW path.
    """

    matches = [(-1, -1)]
    for query_idx, db_idx in zip(*path):
        duplicates = []

        if matches[-1][0] != query_idx:
            if not duplicates:  # empty duplicates
                matches.append((query_idx, db_idx))
            else:
                query_idx = matches[-1][0]

                if cost:
                    best_idx = min(duplicates, key=lambda x: cost[query_idx][x])
                    matches[-1][1] = duplicates[best_idx]
                else:
                    matches[-1][1] = duplicates[len(duplicates) // 2]

                matches.append((query_idx, db_idx))
                duplicates = []
        else:
            duplicates.append(db_idx)
    return matches[1:]


def smooth_frame_intervals(
    lst: List[Tuple[int, int]],
    mv_avg_diff: np.ndarray,
    query_fps: float,
    db_fps: float,
    k: int = 5,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Applies smoothing to frame intervals based on the moving average difference and frame rates.

    Args:
    lst (list): List of tuples containing query and database frame indices. lst should be sorted by query indieces.
    mv_avg_diff (dict): A dictionary holding the moving average of time differences between the query video and the database video, indexed by the frame index of the query video.
    query_fps (float): The frame rate of the query video.
    db_fps (float): The frame rate of the database video.
    k (int, optional): The number of frames to consider for moving average smoothing. Default is 5.

    Returns:
    tuple: A tuple containing the smoothed list and the total difference in frame indices.
    """

    total_diff = 0
    smooth_lst = list()
    N = len(lst)
    for i, (query_idx, db_idx) in enumerate(lst):
        new_db_idx = db_idx
        curr_mv_avg = mv_avg_diff[query_idx]

        # Get next k mv_avg
        next_k_mv_avg = list()
        for ii in range(i + 1, i + k + 1, 1):
            if ii < N:
                next_k_mv_avg.append(mv_avg_diff[lst[ii][0]])

        if len(next_k_mv_avg) != 0:
            avg_of_k = sum(next_k_mv_avg) / len(next_k_mv_avg)

            # only adjust the index when change in mv_avg is consistent
            if abs(curr_mv_avg - avg_of_k) < (2.2 / query_fps):
                new_time = query_idx / query_fps + curr_mv_avg
                new_db_idx = int(new_time * db_fps + 0.5)

        smooth_lst.append((query_idx, new_db_idx))
        total_diff += abs(db_idx - new_db_idx)

    return smooth_lst, total_diff
