"""File discovery and gridding helpers used by GOME-LSIF training."""

import os
import numpy as np


def get_nc_files(folder_path):
    nc_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nc"):
                nc_files.append(os.path.join(root, file))
    return nc_files


def get_first_level_folders(directory):
    folder_paths = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_paths.append(item_path)
    return folder_paths


def calculate_grid_mean(longitude, latitude, data, degree):
    """
    计算每个degree度网格中的数据均值，返回一个二维矩阵，若没有据则为nan
    """
    if np.isnan(data).all():
        # 返回一个合适形状的全nan数组
        return np.full((int(180/degree), int(360/degree)), np.nan)

    # 根据degree创建网格
    grid_data = np.zeros((int(180 / degree), int(360 / degree)))
    grid_count = np.zeros_like(grid_data)

    grid_longitude = np.floor(longitude / degree).astype(int)
    grid_latitude = np.floor(latitude / degree).astype(int)
    grid_longitude[grid_longitude == (180 / degree)] = (180 / degree) - 1
    grid_latitude[grid_latitude == (90 / degree)] = (90 / degree) - 1

    for lon, lat, val in zip(grid_longitude, grid_latitude, data):
        lon_idx = lon + int((180 / degree))
        lat_idx = lat + int((90 / degree))
        grid_data[lat_idx, lon_idx] += val
        grid_count[lat_idx, lon_idx] += 1

    with np.errstate(invalid='ignore'):
        grid_data = np.divide(grid_data, grid_count, where=grid_count != 0)
    grid_data[grid_count == 0] = np.nan
    grid_data = np.flipud(grid_data)
    return grid_data
