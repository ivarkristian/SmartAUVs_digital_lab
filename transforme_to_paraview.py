"""
This script processes chemical data from a NetCDF file, performs spatial interpolation onto a regular grid, 
and then converts the data into VTK format suitable for volume rendering in e.g. ParaView. It handles multiple variables 
and timesteps, ensuring that output directories are created if they do not exist. The script also provides 
functionality to plot the original and interpolated data for comparison.

Functions:
- `compute_interpolation_weights`: Computes interpolation weights for mapping source points to target points.
- `interpolate_values`: Interpolates values based on precomputed weights.
- `transform_netcdf_to_vtk`: Transforms NetCDF data to VTK format.
- `plot_original_vs_interpolated`: Plots original vs. interpolated data for visual comparison.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from xarray.coding.times import SerializationWarning


def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists. If it does not exist, create it.

    Parameters
    ----------
    directory_path : str
        The path to the directory to check/create.

    Returns
    -------
    None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def compute_interpolation_weights(source_points, target_points):
    """
    Compute the interpolation weights for mapping source points to target points.

    Parameters
    ----------
    source_points : ndarray
        The coordinates of the source points (n, 3).
    target_points : ndarray
        The coordinates of the target points (m, 3).

    Returns
    -------
    vertices : ndarray
        The indices of the vertices of the simplex that contains each target point.
    bary : ndarray
        The barycentric coordinates of the target points relative to the simplices.
    """
    tri = Delaunay(source_points)
    simplex = tri.find_simplex(target_points)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = target_points - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate_values(values, vertices, weights, fill_value=np.nan):
    """
    Interpolate values based on precomputed weights.

    Parameters
    ----------
    values : ndarray
        The values at the source points.
    vertices : ndarray
        The indices of the vertices of the simplex that contains each target point.
    weights : ndarray
        The interpolation weights for each target point.
    fill_value : float, optional
        The value to use for points outside the convex hull (default is np.nan).

    Returns
    -------
    ndarray
        The interpolated values at the target points.
    """
    interpolated = np.einsum('nj,nj->n', np.take(values, vertices), weights)
    interpolated[np.any(weights < 0, axis=1)] = fill_value
    return interpolated

def transform_netcdf_to_vtk(data_path, data_variables, resolution, output_dir='vtk_output'):
    """
    Transforms chemical data from a NetCDF file to VTK format suitable for volume rendering.

    This function reads a NetCDF file containing chemical data, interpolates the data onto a regular grid,
    and saves it in VTK format for each time step.

    Parameters
    ----------
    data_path : str
        The file path to the NetCDF file containing the chemical dataset.
    data_variables : list of str
        The names of the data variables to be transformed and saved in VTK format.
    resolution : int
        A multiplication factor for the original resolution of the interpolation grid.
        For example, a value of 2 will double the number of grid points in each spatial dimension.
    output_dir : str, optional
        The directory where the VTK files will be saved (default is 'vtk_output').

    Returns
    -------
    None
    """
    warnings.filterwarnings("ignore", category=SerializationWarning)
    
    # Ensure the output directory exists
    ensure_directory_exists(output_dir)

    # Load the NetCDF file
    dataset = xr.open_dataset(data_path, decode_times=False)

    # Convert ITIME to actual timestamps
    time_reference = pd.Timestamp('1970-01-01')
    times_in_seconds = dataset['Itime2'].values / 1000
    converted_times = time_reference + pd.to_timedelta(times_in_seconds, unit='s')
    dataset['time'] = ('time', converted_times)

    # Crop data to exclude boundary mesh (value 72710 chosen by inspecting the data)
    x_coords = dataset['x'].values[:72710]
    y_coords = dataset['y'].values[:72710]
    z_coords = dataset['siglay'].values[:, :72710]

    # Define target grid for interpolation
    x_min = np.floor(np.min(x_coords))
    x_max = np.ceil(np.max(x_coords))
    y_min = np.floor(np.min(y_coords))
    y_max = np.ceil(np.max(y_coords))
    z_min = np.floor(np.min(z_coords))
    z_max = np.ceil(np.max(z_coords))

    x_target = np.linspace(x_min, x_max, resolution * int(x_max - x_min) + 1)
    y_target = np.linspace(y_min, y_max, resolution * int(y_max - y_min) + 1)
    
    X_target, Y_target = np.meshgrid(x_target, y_target)
    z_target = np.linspace(z_min, z_max, z_coords.shape[0])

    dims = (len(x_target), len(y_target), len(z_target))
    spacing = (x_target[1] - x_target[0], y_target[1] - y_target[0], z_target[1] - z_target[0])

    source_points = np.column_stack((x_coords, y_coords))
    target_points = np.column_stack((X_target.ravel(), Y_target.ravel()))

    # Precompute interpolation weights
    vertices, weights = compute_interpolation_weights(source_points, target_points)

    for time_index in range(len(dataset['time'].values)):
        interpolated_data = {var: np.zeros((len(x_target), len(y_target), len(z_target))) for var in data_variables}

        for layer_index in range(z_coords.shape[0]):
            print(f"Processing time index: {time_index}/{len(dataset['time'].values) - 1}, "
                  f"layer index: {layer_index}/{z_coords.shape[0] - 1}")

            for data_variable in data_variables:
                data_sample = dataset[data_variable].isel(time=time_index, siglay=layer_index).values[:72710]
                interpolated_layer = interpolate_values(data_sample, vertices, weights).reshape(len(x_target), len(y_target))

                interpolated_data[data_variable][:, :, layer_index] = interpolated_layer

        for data_variable in data_variables:
            interpolated_flat = interpolated_data[data_variable].ravel(order='F')

            # Create VTK image data object
            image_data = vtk.vtkImageData()
            image_data.SetDimensions(dims)
            image_data.SetSpacing(spacing)
            image_data.SetOrigin(np.min(x_target), np.min(y_target), np.min(z_target))

            # Convert the interpolated data to VTK format
            vtk_data = numpy_to_vtk(interpolated_flat, deep=True)
            vtk_data.SetName(f'Interpolated_{data_variable}')

            image_data.GetPointData().SetScalars(vtk_data)

            # Write VTK file for the current time step
            output_file = os.path.join(output_dir, f"interpolated_{data_variable}_timestep_{time_index}.vti")
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(output_file)
            writer.SetInputData(image_data)
            writer.Write()

    print("VTK files saved for each time step suitable for volume rendering.")

def plot_original_vs_interpolated(data_path, data_variable, timestep, layer):
    """
    Plots the original and interpolated data variables side by side for a given timestep and layer.

    Parameters
    ----------
    data_path : str
        The file path to the NetCDF file containing the chemical dataset.
    data_variable : str
        The name of the data variable to be plotted.
    timestep : int
        The index of the timestep to be plotted.
    layer : int
        The index of the layer to be plotted.

    Returns
    -------
    None
    """
    warnings.filterwarnings("ignore", category=SerializationWarning)

    # Load the NetCDF file
    dataset = xr.open_dataset(data_path, decode_times=False)

    # Convert ITIME to actual timestamps if necessary
    if 'Itime' in dataset and 'Itime2' in dataset:
        time_reference = pd.Timestamp('1970-01-01')
        times_in_seconds = dataset['Itime2'].values / 1000
        converted_times = time_reference + pd.to_timedelta(times_in_seconds, unit='s')
        dataset['time'] = ('time', converted_times)
    
    # Extract coordinates and data
    x_coords = dataset['x'].values[:72710]
    y_coords = dataset['y'].values[:72710]
    data_sample = dataset[data_variable].isel(time=timestep, siglay=layer).values[:72710]

    print(f'Original data range: {np.min(data_sample)} to {np.max(data_sample)}')
    print(f'Original data mean: {np.mean(data_sample)}')
    print(f'Original data standard deviation: {np.std(data_sample)}')

    # Define target grid for interpolation
    x_target = np.linspace(np.floor(np.min(x_coords)), np.ceil(np.max(x_coords)), 
                           2 * int(np.ceil(np.max(x_coords)) - np.floor(np.min(x_coords))) + 1)
    y_target = np.linspace(np.floor(np.min(y_coords)), np.ceil(np.max(y_coords)), 
                           2 * int(np.ceil(np.max(y_coords)) - np.floor(np.min(y_coords))) + 1)
    X_target, Y_target = np.meshgrid(x_target, y_target)

    # Perform interpolation
    interpolated_layer = griddata((x_coords, y_coords), data_sample, (X_target, Y_target), method='cubic')

    print(f'Interpolated data range: {np.min(interpolated_layer)} to {np.max(interpolated_layer)}')
    print(f'Interpolated data mean: {np.mean(interpolated_layer)}')
    print(f'Interpolated data standard deviation: {np.std(interpolated_layer)}')

    # Plot original data
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_coords, y_coords, c=data_sample, cmap='viridis')
    plt.title(f'Original {data_variable} at timestep {timestep}, layer {layer}')
    plt.colorbar(label=data_variable)
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot interpolated data
    plt.subplot(1, 2, 2)
    plt.contourf(X_target, Y_target, interpolated_layer, cmap='viridis')
    plt.title(f'Interpolated {data_variable} at timestep {timestep}, layer {layer}')
    plt.colorbar(label=data_variable)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_path = "SMART-AUVs_OF-June-1a-0001.nc"

    data_variables = ['pH', 'pCO2']
    
    # Specify the output directory for VTK files
    output_directory = "vtk_output"

    transform_netcdf_to_vtk(data_path, data_variables, resolution=2, output_dir=output_directory)
