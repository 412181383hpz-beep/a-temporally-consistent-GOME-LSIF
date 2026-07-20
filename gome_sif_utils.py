"""Shared utilities used by the GOME SIF production scripts."""

import os

import numpy as np
from sklearn.decomposition import PCA


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
    """Calculate the mean value in each latitude-longitude grid cell."""
    if np.isnan(data).all():
        return np.full((int(180 / degree), int(360 / degree)), np.nan)

    grid_data = np.zeros((int(180 / degree), int(360 / degree)))
    grid_count = np.zeros_like(grid_data)

    grid_longitude = np.floor(longitude / degree).astype(int)
    grid_latitude = np.floor(latitude / degree).astype(int)
    grid_longitude[grid_longitude == (180 / degree)] = (180 / degree) - 1
    grid_latitude[grid_latitude == (90 / degree)] = (90 / degree) - 1

    for lon, lat, val in zip(grid_longitude, grid_latitude, data):
        lon_idx = lon + int(180 / degree)
        lat_idx = lat + int(90 / degree)
        grid_data[lat_idx, lon_idx] += val
        grid_count[lat_idx, lon_idx] += 1

    with np.errstate(invalid="ignore"):
        grid_data = np.divide(grid_data, grid_count, where=grid_count != 0)
    grid_data[grid_count == 0] = np.nan
    return np.flipud(grid_data)


def lon_lat_arange(degree):
    """Return grid-cell-centre longitude and latitude sequences."""
    start = -180 + degree / 2
    end = 180 - degree / 2
    lon_sequence = np.arange(start, end + degree, degree)

    start = 90 - degree / 2
    end = -90 + degree / 2
    lat_sequence = np.arange(start, end - degree, -degree)
    return lon_sequence, lat_sequence


def sif_retrieval(poly_i, rad_i, free):
    """Retrieve SIF, reduced chi-square, and SIF variance by least squares."""
    rad_mean = np.mean(rad_i)
    result, residuals, rank, singular_values = np.linalg.lstsq(
        poly_i, rad_i, rcond=None
    )
    rad_rebuilt = np.dot(poly_i, result)
    snr = rad_mean / np.std(rad_i - rad_rebuilt)
    chi_square = np.sum((rad_rebuilt - rad_i) ** 2 / (rad_i / snr) ** 2)
    reduced_chi_square = chi_square / (len(rad_i) - free)
    sif = result[-1]

    residual_sum = np.sum((rad_i - rad_rebuilt) ** 2)
    residual_variance = residual_sum / (len(rad_i) - free)
    covariance = residual_variance * np.linalg.inv(poly_i.T @ poly_i)
    sif_variance = covariance[-1, -1]
    return sif, reduced_chi_square, sif_variance


def filter_outliers_with_pca(data, n_components=20, error_threshold=0.005):
    """Remove spectra whose maximum PCA reconstruction error exceeds a threshold."""
    pca = PCA(n_components=n_components)
    pca.fit(data)

    reduced_data = pca.transform(data)
    reconstructed_data = pca.inverse_transform(reduced_data)
    absolute_difference = np.abs(data - reconstructed_data)
    relative_error = absolute_difference / (np.abs(data) + 1e-10)
    maximum_relative_error = np.max(relative_error, axis=1)
    return data[maximum_relative_error < error_threshold]
