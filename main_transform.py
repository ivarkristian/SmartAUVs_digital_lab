# %%
import os
import importlib
import chem_utils
import transform_to_paraview

# %%
# Setup scenario data directories
data_dir = '../scenario_1c_medium/'
#data_dir = '../scenario_1f_small/'
resolution = 1
output_dir = data_dir + 'vtk_output_test'

# %%
# Read and clean list of .nc files
files = os.listdir(data_dir)
# Create a new list with strings that end with '.nc'
nc_files = [s for s in files if s.endswith('.nc')]
nc_files.sort()
print(f'Files of type .nc:\n{nc_files}')

# %%
# Load the dataset and print attributes
data_path = data_dir + nc_files[0]
dataset = chem_utils.load_chemical_dataset(data_path)
dataset

# %%
# Testing u, v -> w, d conversion
importlib.reload(transform_to_paraview)
data_vars = ['pH', 'u', 'v']
data_path = data_dir + nc_files[0]
transform_to_paraview.transform_netcdf_to_vtk(data_path, data_vars, resolution, output_dir)

# %%
# Convert the whole scenario
importlib.reload(transform_to_paraview)
# Typical data_vars for CO2 release
# data_vars = ['salinity', 'temp', 'u', 'v', 'ww', 'DIC', 'dDIC', 'pH', 'dpH', 'pCO2', 'dpCO2']
# Typical data_vars for CH4 release
data_vars = ['salinity', 'temp', 'u', 'v', 'ww', 'DYE']

for file in nc_files:
    data_path = data_dir + file
    transform_to_paraview.transform_netcdf_to_vtk(data_path, data_vars, resolution, output_dir)

# %%
