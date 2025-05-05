# %%
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import Arc, FancyArrowPatch
#import seaborn as sns
import math
import numpy as np
import pandas as pd
import torch
import gpytorch
import pickle_reader
import lawnmower_path as lp
import path
import path_utils

# Disable LaTeX renderingß to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
directory_path_fig1 = '/Users/ikw/code/out_files/fig1_run'
environments = pickle_reader.collect_data_from_directory(directory_path_fig1 + '/figures/environments', '.pickle')
# Plot two environments, anisotropies 1.0 and 180
# Assuming 'environments' is your list of figures
anisotropies = [1.13, 10.76]
titles = []
for i, anisotropy in enumerate(anisotropies):
    titles.append(f'Aspect ratio = {anisotropy}')

# Initialize variables to collect image data and determine global vmin and vmax
# Remove color bars and save figures to images
image_list = []
for i, fig in enumerate(environments):
    ax = fig.axes[0]
    ax.set_title('')  # Remove existing title
    ax.set_title(titles[i], fontsize=16)  # Set new title

    # Remove the color bar if it exists
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[1]
        fig.delaxes(cbar_ax)
        fig.subplots_adjust(right=0.9)

    # Save figure to image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img = mpimg.imread(buf)
    buf.close()
    image_list.append(img)

# Create a new figure with 2x2 subplots
fig_combined, axes = plt.subplots(1, 2, figsize=(10, 4))
axes = axes.flatten()

# Display images in subplots
for i, (img, ax) in enumerate(zip(image_list, axes)):
    ax.imshow(img)
    ax.axis('off')
    #ax.set_title(titles[i], fontsize=12)

#fig_combined.suptitle('Example generated emissions', fontsize=16, fontweight='bold', x=0.4, y=0.98)

# Adjust layout to minimize space between subplots and make space for the color bar
fig_combined.subplots_adjust(
#    left=0.05,   # Reduce left margin
#    right=0.75,  # Reduce right margin
#    bottom=0.12, # Reduce bottom margin to make space for color bar
#    top=0.92,    # Slightly reduce top margin
    wspace=-0.2, # Minimize horizontal space between subplots
#    hspace=0.01  # Minimize vertical space between subplots
)

# Add a single color bar at the bottom with reduced width
cbar_ax = fig_combined.add_axes([0.43, 0.10, 0.2, 0.03])  # [left, bottom, width, height]
norm = colors.Normalize(vmin=0, vmax=1.0)
sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
fig_combined.colorbar(
    sm,
    cax=cbar_ax,
    orientation='horizontal',
    label='Intensity',
    ticklocation='bottom'
)

# Optionally, adjust color bar tick parameters
cbar_ax.tick_params(labelsize=8)
#fig_combined.tight_layout(w_pad=-12.0)

# Save the figure as an EPS file with high DPI
fig_combined.savefig('figures/' + 'example_emissions.eps', format='eps', dpi=300)

# Display the combined figure
plt.show()

# %%
# Prediction figure
# %%
directory_path_fig1 = '/Users/ikw/code/out_files/fig1_run'
data_list = pickle_reader.collect_data_from_directory(directory_path_fig1, '.pickle')
predictions = pickle_reader.collect_data_from_directory(directory_path_fig1 + '/figures', '.pickle')
df = pd.DataFrame(data_list[4:])

# %%
# Assuming 'environments' is your list of figures
anisotropies = [1, 180]
titles = ['Plain pattern, SE kernel (RMSE = 0.08)', 'Plain pattern, SE-ARD kernel (RMSE = 0.06)', 'Grid pattern, SE kernel (RMSE = 0.05)', 'Grid pattern, SE-ARD kernel (RMSE = 0.02)']
#for i, anisotropy in enumerate(anisotropies):
#    titles.append(f'Anisotropy = {anisotropy}')

# Initialize variables to collect image data and determine global vmin and vmax
# Remove color bars and save figures to images
image_list = []
for i, fig in enumerate(predictions[4:]):
    ax = fig.axes[0]
    ax.set_title('')  # Remove existing title
    ax.set_title(titles[i], fontsize=16)  # Set new title

    # Remove the color bar if it exists
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[1]
        fig.delaxes(cbar_ax)
        fig.subplots_adjust(right=0.9)

    # Save figure to image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img = mpimg.imread(buf)
    buf.close()
    image_list.append(img)

# Create a new figure with 2x2 subplots
fig_combined, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# Display images in subplots
for i, (img, ax) in enumerate(zip(image_list, axes)):
    ax.imshow(img)
    ax.axis('off')
    #ax.set_title(titles[i], fontsize=12)

fig_combined.suptitle('Environment predictions', fontsize=16, fontweight='bold', x=0.4, y=0.96)

# Adjust layout to minimize space between subplots and make space for the color bar
fig_combined.subplots_adjust(
    left=0.05,   # Reduce left margin
    right=0.75,  # Reduce right margin
    bottom=0.12, # Reduce bottom margin to make space for color bar
    top=0.92,    # Slightly reduce top margin
    wspace=-0.04, # Minimize horizontal space between subplots
    hspace=0.01  # Minimize vertical space between subplots
)

# Add a single color bar at the bottom with reduced width
cbar_ax = fig_combined.add_axes([0.2, 0.08, 0.4, 0.03])  # [left, bottom, width, height]
norm = colors.Normalize(vmin=0, vmax=1.0)
sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
fig_combined.colorbar(
    sm,
    cax=cbar_ax,
    orientation='horizontal',
    label='Intensity',
    ticklocation='bottom'
)

# Optionally, adjust color bar tick parameters
cbar_ax.tick_params(labelsize=8)

# Save the figure as an EPS file with high DPI
fig_combined.savefig('figures/' + 'example_prdictions.eps', format='eps', dpi=300)

# Display the combined figure
plt.show()

# %%
# Explanatory figure
def plot_env(x, y, values, path=None, vmin=None, vmax=None, title='No title'):
    fig, ax = plt.subplots(figsize=(8, 8))
    if vmin == None:
        vmin = values.min()
    if vmax == None:
        vmax = values.max()

    scatter = ax.scatter(x, y, color='white', s=1, vmin=vmin, vmax=vmax)
    #cbar = fig.colorbar(scatter, ax=ax)
    #cbar.set_label('Value')

    if path is not None:
        ax.scatter(path[:, 0], path[:, 1], color='black', s=1, label='Path')
    
    # Add labels and title
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.close()
    return fig

# %%
# Create the figure
# Define size and resolution of environment
scenario_x_min = 0
scenario_x_max = 250
scenario_y_min = 0
scenario_y_max = 250
sc_len_x = scenario_x_max - scenario_x_min
sc_len_y = scenario_y_max - scenario_y_min

# Create environment x and y locations
env_x = np.linspace(scenario_x_min, scenario_x_max, sc_len_x, dtype=np.float64)
env_y = np.linspace(scenario_y_min, scenario_y_max, sc_len_y, dtype=np.float64)
env_xy = torch.tensor(np.column_stack((env_x, env_y)), dtype=torch.float64)
env_xy = gpytorch.utils.grid.create_data_from_grid(env_xy)
x = env_xy[:, 0]
y = env_xy[:, 1]

sc_len_x = scenario_x_max - scenario_x_min
sc_len_y = scenario_y_max - scenario_y_min
path_len = sc_len_x/2.0 * np.sqrt(2.0)
path_delta = (sc_len_x - path_len)/2.0

path_x_min = scenario_x_min + path_delta
path_x_max = scenario_x_max - path_delta
path_y_min = scenario_y_min + path_delta
path_y_max = scenario_y_max - path_delta

resolution = 3000
x_data = np.linspace(path_x_min, path_x_max, resolution, dtype=np.float64)
y_data = np.linspace(path_y_min, path_y_max, resolution, dtype=np.float64)

spacing = 20
# Generate waypoints for the lawnmower path with specified parameters 
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=spacing, min_turn_radius=int(spacing/2), siglay=0, direction='x'
)

# remove duplicate waypoints
waypoints = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)

ts = np.datetime64('1970-01-01T13:00:00.000000000')
speed = 1.5
sample_freq = 1
plain_sample_coords_list = path.path(waypoints, ts, speed, sample_freq, synoptic=True)

# Convert the sample_coords list of tuples to a NumPy array
sample_coords_no_rot = np.array([tup[:2] for tup in plain_sample_coords_list])

# Compute means, min and max for x and y axes
wp_x = waypoints[:, 0]
wp_y = waypoints[:, 1]

wp_x_min, wp_x_max = np.min(wp_x), np.max(wp_x)
wp_y_min, wp_y_max = np.min(wp_y), np.max(wp_y)
x_mean = wp_x_min + (wp_x_max - wp_x_min)/2.0
y_mean = wp_y_min + (wp_y_max - wp_y_min)/2.0

xy_to_rotate = torch.tensor(sample_coords_no_rot[:, :2] - [x_mean, y_mean], dtype=torch.float64)

rot_dir = -30
xy_rot = path_utils.rotate_points(xy_to_rotate, rot_dir) + torch.tensor([x_mean, y_mean], dtype=torch.float64)

# %%
fig = plot_env(x, y, [0]*len(x), xy_rot, vmin=0, vmax=0, title='Plain pattern geometry definitions')
ax = fig.gca()

center = (xy_rot[-1, 0], xy_rot[:, 1].min())
radius = 50
theta1 = 0
theta2 = 60
# Create the arc
arc = Arc(center, width=2*radius, height=2*radius,
            angle=0, theta1=theta1, theta2=theta2,
            color='black', linewidth=1)

# Add the arc to the axes
ax.add_patch(arc)

# Annotate the angle
ax.text(center[0]+15, center[1]+10, r'$\delta = 60^\circ$',
        fontsize=12, color='black', ha='left', va='center')

# Add arrow
arrow_start = (25, center[1])
arrow_end = (225, center[1])

# Create current arrow patch
arrow = FancyArrowPatch(
    posA=(arrow_start[0], arrow_start[1]),
    posB=(arrow_end[0], arrow_end[1]),
    arrowstyle='->',
    color='black',
    linewidth=1,
    mutation_scale=40  # Adjust size of the arrowhead
)

# Add the arrow to the axes
ax.add_patch(arrow)
ax.text(sc_len_x/2.0, arrow_start[1]-10, r'Current direction ($\alpha = 0^\circ$)',
        fontsize=12, color='black', ha='center', va='center')

# Create spacing arrow patch
r = 30.0*math.pi/180.0
arr_start = (72, 190)
arr_end = (arr_start[0] + 23*math.cos(r), arr_start[1] - 23*math.sin(r))
arrow = FancyArrowPatch(
    posA=arr_start,
    posB=arr_end,
    arrowstyle='<->',
    color='black',
    linewidth=1,
    mutation_scale=20  # Adjust size of the arrowhead
)

# Add the arrow to the axes
ax.add_patch(arrow)
ax.text(arr_start[0] - 25, arr_start[1], r'($S = 20m$)',
        fontsize=12, color='black', ha='center', va='center')

# Save the figure as an EPS file with high DPI
fig.savefig('figures/' + 'geometry_definitions.eps', format='eps', dpi=300)
fig

# %%
