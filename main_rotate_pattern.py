# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import lawnmower_path as lp
import path_utils
import chem_utils
import path

# Disable LaTeX renderingß to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
# Define size and resolution of environment
scenario_x_min = 0
scenario_x_max = 250
scenario_y_min = 0
scenario_y_max = 250

sc_len = scenario_x_max - scenario_x_min
path_len = sc_len/2.0 * np.sqrt(2.0)
path_delta = (sc_len - path_len)/2.0

path_x_min = scenario_x_min + path_delta
path_x_max = scenario_x_max - path_delta
path_y_min = scenario_y_min + path_delta
path_y_max = scenario_y_max - path_delta

resolution = 3000
x_data = np.linspace(path_x_min, path_x_max, resolution)
y_data = np.linspace(path_y_min, path_y_max, resolution)
depth = 67

# %%
# Generate waypoints for the lawnmower path with specified parameters 
grid_spacings = [10, 20]

waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=20, min_turn_radius=5, siglay=depth, direction='x'
)

# remove duplicate waypoints
waypoints_with_turns = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)

# %%
# Visualize the path with the domain background and labeled waypoints
wp_x = waypoints_with_turns[:, 0]
wp_y = waypoints_with_turns[:, 1]

wp_x_min, wp_x_max = np.min(wp_x), np.max(wp_x)
wp_y_min, wp_y_max = np.min(wp_y), np.max(wp_y)

x_mean = wp_x_min + (wp_x_max - wp_x_min)/2.0
y_mean = wp_y_min + (wp_y_max - wp_y_min)/2.0

x_min = x_mean - (x_mean - np.min(wp_x))*np.sqrt(2.0)
y_min = y_mean - (y_mean - np.min(wp_y))*np.sqrt(2.0)
x_max = x_mean + (np.max(wp_x) - x_mean)*np.sqrt(2.0)
y_max = y_mean + (np.max(wp_y) - y_mean)*np.sqrt(2.0)

lp.scatter_plot_points_and_path(waypoints_with_turns[:, :2], x_min, x_max, y_min, y_max)

# %%
# Load data set from path
data_dir = '../scenario_1c_medium/'
# Read and clean list of .nc files
files = os.listdir(data_dir)
# Create a new list with strings that end with '.nc'
nc_files = [s for s in files if s.endswith('.nc')]
nc_files.sort()
print(f'Files of type .nc:\n{nc_files}')

file_num = 7
data_file = data_dir + nc_files[file_num]
dataset = chem_utils.load_chemical_dataset(data_file)


# %%
# Read and plot depths from dataset
ts = 4
data_parameter = 'pH'

for ts in range(0, 12):
    for depth in range(66, 69):
        val_dataset = dataset[data_parameter].isel(time=ts, siglay=depth)
        val = val_dataset.values[:72710]
        x = val_dataset['x'].values[:72710]
        y = val_dataset['y'].values[:72710]
        x = x - x.min()
        y = y - y.min()
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(y, x, c=val, cmap='coolwarm', s=2, vmin=val.min(), vmax=val.max())
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Value')

        # Add labels and title
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_title(f'TS {file_num*12 + ts}, {data_parameter} at {depth}m depth')

        plt.show()

# %%
# Get current
data_parameter = 'u'

for ts in range(4, 5):
    for depth in range(67, 68):
        val_dataset = dataset[data_parameter].isel(time=ts, siglay=depth)
        val = val_dataset.values[:72710]
        #x = val_dataset['x'].values[:72710]
        #y = val_dataset['y'].values[:72710]
        #x = x - x.min()
        #y = y - y.min()
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y, c=val, cmap='coolwarm', s=2, vmin=val.min(), vmax=val.max())
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
# Extract using interpolating extract function path.path()
start_time = dataset['time'].values[4]
speed = 1.5
waypoints = waypoints_with_turns
sample_freq = 1
measurements, sample_coords = path.path(dataset, waypoints, start_time, speed, sample_freq, data_variable='pH', synoptic=True)

# %%
# Plot it
vminmax = [min(measurements), max(measurements)]
fig = path.plot(waypoints_with_turns, sample_coords, measurements, 'pH [m]', vminmax=vminmax)
fig.show()

# %%
# Convert the list of tuples to a NumPy array
filtered_data = [tup[:3] for tup in sample_coords]
data_array = np.array(filtered_data)

# %%
# Rotate path
angle_deg = 45.0

# Subtract means
x_off = (wp_x_max - wp_x_min)/2.0
y_off = (wp_y_max - wp_y_min)/2.0

z_coords = waypoints_with_turns[:, 2:]
wp_to_rotate = waypoints_with_turns[:, :2] - [x_mean, y_mean]
wp_rotated = path_utils.rotate_points(wp_to_rotate, angle_deg) + [x_mean, y_mean]
waypoints_with_turns_rotated = np.hstack((wp_rotated, z_coords))

# New domain boundaries for the plot
#x_min, x_max = np.min(x_data), np.max(x_data)
#y_min, y_max = np.min(y_data), np.max(y_data)
lp.scatter_plot_points_and_path(waypoints_with_turns_rotated[:, :2], x_min, x_max, y_min, y_max)


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