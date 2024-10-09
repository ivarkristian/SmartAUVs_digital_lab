# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import lawnmower_path as lp
import path_utils
import chem_utils
import path

# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
x_data = np.linspace(0, 250, 3000)
y_data = np.linspace(0, 250, 3000)
depth = 67
# Generate waypoints for the lawnmower path with specified parameters
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=5, min_turn_radius=5, siglay=depth, direction='x'
)

# remove duplicate waypoints
waypoints_with_turns = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)


# %%
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
# Read and plot depths from dataset
ts = 4
data_parameter = 'pH'

for ts in range(10, 12):
    for depth in range(67, 68):
        val_dataset = dataset[data_parameter].isel(time=ts, siglay=depth)
        val = val_dataset.values[:72710]
        x = val_dataset['x'].values[:72710]
        y = val_dataset['y'].values[:72710]
        x = x - x.min()
        y = y - y.min()
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Value')

        # Add labels and title
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_title(f'TS {ts}, {data_parameter} at {depth}m depth')

        plt.show()

# %%
importlib.reload(path)
importlib.reload(chem_utils)
importlib.reload(lp)

# %%
# Extract using interpolating extract function
#start_time = '2020-01-01T00:00:00.000000000'
start_time = dataset['time'].values[4]
speed = 10
way_points = waypoints_with_turns
sample_freq = 10
threshold = np.inf
pattern = None
measurements, sample_coords = path.path(dataset, None, start_time, speed, way_points, sample_freq, threshold, pattern, 'pH')

#metadata = (100, 100, 69, 3, 5)

# %%
fig = path.plot(waypoints_with_turns, sample_coords, measurements, 'pH', 'Title')
fig.show()

# %%
# Decide scenario, depth, ts
# Decide grid density
# Compute waypoints
# Decide rotation
# Rotate
# Sample (synoptic)
# Normalize sample values to [0 1]
# Train GP
# Predict
# Compute RMS error
# Correlate RMSE with grid density, rotation.