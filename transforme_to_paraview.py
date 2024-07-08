import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import warnings
from xarray.coding.times import SerializationWarning
import matplotlib.pyplot as plt

def transform_netCDF_to_vtk(data_path, data_variable):
    """
    Transforms chemical data from a NetCDF file to VTK format suitable for volume rendering.

    This function reads a NetCDF file containing chemical data, interpolates the data onto a regular grid,
    and saves it in VTK format for each time step.

    Parameters
    ----------
    data_path : str
        The file path to the NetCDF file containing the chemical dataset.
    data_variable : str
        The name of the data variable to be transformed and saved in VTK format.

    Returns
    -------
    None
    """
    warnings.filterwarnings("ignore", category=SerializationWarning)

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
    x_target = np.linspace(np.floor(np.min(x_coords)), np.ceil(np.max(x_coords)), 
                           2 * int(np.ceil(np.max(x_coords)) - np.floor(np.min(x_coords))) + 1)
    y_target = np.linspace(np.floor(np.min(y_coords)), np.ceil(np.max(y_coords)), 
                           2 * int(np.ceil(np.max(y_coords)) - np.floor(np.min(y_coords))) + 1)
    X_target, Y_target = np.meshgrid(x_target, y_target)
    z_target = np.linspace(np.floor(np.min(z_coords)), np.ceil(np.max(z_coords)), z_coords.shape[0])

    dims = (len(x_target), len(y_target), len(z_target))
    spacing = (x_target[1] - x_target[0], y_target[1] - y_target[0], z_target[1] - z_target[0])

    source_points = np.column_stack((x_coords, y_coords))

    for time_index in range(len(dataset['time'].values)):
        interpolated_data_all_layers = np.zeros((len(x_target), len(y_target), len(z_target)))

        for layer_index in range(z_coords.shape[0]):
            print(f"Time index: {time_index}/{len(dataset['time'].values) - 1}, Layer index: {layer_index}/{z_coords.shape[0] - 1}")

            data_sample = dataset[data_variable].isel(time=time_index, siglay=layer_index).values[:72710]

            interpolated_layer = griddata(source_points, data_sample, (X_target, Y_target), method='linear')

            if interpolated_layer is not None:
                interpolated_data_all_layers[:, :, layer_index] = interpolated_layer
            else:
                # Handle case where interpolation fails
                interpolated_data_all_layers[:, :, layer_index] = np.nan

        # Flatten the interpolated data in Fortran order to match VTK's expectations
        interpolated_data_flat = interpolated_data_all_layers.ravel(order='F')

        # Create VTK image data object
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(dims)
        image_data.SetSpacing(spacing)
        image_data.SetOrigin(np.min(x_target), np.min(y_target), np.min(z_target))

        # Convert the interpolated data to VTK format
        vtk_interpolated_data = numpy_to_vtk(interpolated_data_flat, deep=True)
        vtk_interpolated_data.SetName(f'Interpolated_{data_variable}')

        image_data.GetPointData().SetScalars(vtk_interpolated_data)

        # Write VTK file for the current time step
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(f"vtks/new/interpolated_{data_variable}_timestep_{time_index}.vti")
        writer.SetInputData(image_data)
        writer.Write()

    print("VTK files saved for each time step suitable for volume rendering.")

if __name__ == '__main__':
    data_path = "utils_and_data/plume_data/SMART-AUVs_OF-June-1a-0001.nc"
    data_variable = 'pH'
    transform_netCDF_to_vtk(data_path, data_variable)
