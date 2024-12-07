'''
Compute the minimum distance between meshes at certain states
'''

import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
from assets.transform import get_transform_matrix, transform_pts_by_matrix

# Profiling code
import cProfile
import pstats
from functools import wraps
import time


def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            print(f"\nProfile for {func.__name__}:")
            stats.print_stats(20)

    return wrapper


def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper



def compute_all_mesh_distance(meshes, states):
    '''
    Compute the minimum distance between meshes at certain states
    '''
    assert len(meshes) == len(states)

    mats, inv_mats = [], []
    for i in range(len(meshes)):
        mat = get_transform_matrix(states[i])
        mats.append(mat)
        inv_mats.append(np.linalg.inv(mat))

    d = np.inf
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            v_i_trans = transform_pts_by_matrix(meshes[i].vertices.T, inv_mats[j].dot(mats[i]))
            v_j_trans = transform_pts_by_matrix(meshes[j].vertices.T, inv_mats[i].dot(mats[j]))
            d_ij = meshes[i].min_distance(v_j_trans.T)
            d_ji = meshes[j].min_distance(v_i_trans.T)
            d = np.min([d, d_ij, d_ji])
    
    return d


@time_function
def compute_move_mesh_distance(move_mesh, still_meshes, state):
    '''
    Compute the minimum distance between meshes at certain states
    '''
    move_mat = get_transform_matrix(state)
    move_inv_mat = np.linalg.inv(move_mat)

    v_m_trans = transform_pts_by_matrix(move_mesh.vertices.T, move_mat).T

    d = np.inf
    for i in range(len(still_meshes)):
        v_s_trans = transform_pts_by_matrix(still_meshes[i].vertices.T, move_inv_mat).T
        d_ms = move_mesh.min_distance(v_s_trans)
        d_sm = still_meshes[i].min_distance(v_m_trans)
        d = np.min([d, d_ms, d_sm])

    return d
