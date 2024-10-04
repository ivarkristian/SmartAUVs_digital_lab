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

# Generate waypoints for the lawnmower path with specified parameters
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=50, min_turn_radius=50, siglay=1, direction='x'
)

# remove duplicate waypoints
new_wps = lp.remove_consecutive_duplicate_wps(waypoints_with_turns)


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
ts = 3
data_parameter = 'pH'

for depth in range(55, 70):
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
    ax.set_title(f'{data_parameter} at {depth}m depth')

    plt.show()

# %%
importlib.reload(path)
importlib.reload(chem_utils)
importlib.reload(lp)

# %%
# Extract using interpolating extract function
#start_time = '2020-01-01T00:00:00.000000000'
start_time = dataset['time'].values[0]
speed = 100
way_points = waypoints_with_turns
sample_freq = 100
threshold = np.inf
pattern = None
measurements, sample_coords = path.path(dataset, None, start_time, speed, way_points, sample_freq, threshold, pattern, )

#metadata = (100, 100, 69, 3, 5)
# %%
