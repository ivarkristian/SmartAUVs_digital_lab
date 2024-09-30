# %%
import os
import numpy as np
import lawnmower_path as lp
import path_utils
import chem_utils

# %%
x_data = np.linspace(0, 500, 3000)
y_data = np.linspace(0, 500, 3000)

# Generate waypoints for the lawnmower path with specified parameters
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=50, min_turn_radius=50, siglay=1, direction='x'
)

# Domain boundaries for the plot
x_min, x_max = np.min(x_data), np.max(x_data)
y_min, y_max = np.min(y_data), np.max(y_data)

# Visualize the path with the domain background and labeled waypoints
lp.scatter_plot_points_and_path(waypoints_with_turns[:, :2], x_min, x_max, y_min, y_max)

# %%
# Rotate waypoints
angle_deg = 45.0

# Subtract means
x_off = (x_max - x_min)/2.0
y_off = (y_max - y_min)/2.0

z_coords = waypoints_with_turns[:, 2:]
wp_to_rotate = waypoints_with_turns[:, :2] - [x_off, y_off]
wp_rotated = path_utils.rotate_points(wp_to_rotate, angle_deg) + [x_off, y_off]
waypoints_with_turns_rotated = np.hstack((wp_rotated, z_coords))

# New domain boundaries for the plot
x_min, x_max = np.min(x_data), np.max(x_data)
y_min, y_max = np.min(y_data), np.max(y_data)
lp.scatter_plot_points_and_path(waypoints_with_turns_rotated[:, :2], x_min, x_max, y_min, y_max)

# %%
# Load data set from path
data_dir = '../scenario_1c_medium/'
# Read and clean list of .nc files
files = os.listdir(data_dir)
# Create a new list with strings that end with '.nc'
nc_files = [s for s in files if s.endswith('.nc')]
nc_files.sort()
print(f'Files of type .nc:\n{nc_files}')

data_file = data_dir + nc_files[0]
dataset = chem_utils.load_chemical_dataset(data_file)

# %%
# Create sampling points


# %%
# Sample from dataset
