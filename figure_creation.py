# %%
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg
import seaborn as sns
import pickle_reader

# %%
directory_path_fig1 = '/Users/ikw/code/out_files/fig1_run'
environments = pickle_reader.collect_data_from_directory(directory_path_fig1 + '/figures/environments', '.pickle')
# Plot two environments, anisotropies 1.0 and 180
# Assuming 'environments' is your list of figures
anisotropies = [1, 180]
titles = []
for i, anisotropy in enumerate(anisotropies):
    titles.append(f'Anisotropy = {anisotropy}')

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

fig_combined.suptitle('Example generated emissions', fontsize=16, fontweight='bold', x=0.4, y=0.98)

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
fig_combined.savefig('figures/' + 'example_emissions.eps', format='eps', dpi=300)

# Display the combined figure
plt.show()

# %%
# Prediction figure
# %%
directory_path_fig1 = '/Users/ikw/code/out_files/fig1_run'
data_list = pickle_reader.collect_data_from_directory(directory_path, '.pickle')
predictions = pickle_reader.collect_data_from_directory(directory_path_fig1 + '/figures', '.pickle')
df = pd.DataFrame(data_list[4:])

# %%
# Assuming 'environments' is your list of figures
anisotropies = [1, 180]
titles = ['Plain pattern, SE kernel (RMSE = 0.08)', 'Plain pattern, SE-ARD kernel (RMSE = 0.06)', 'Grid pattern, SE kernel (RMSE = 0.05)', 'Plain pattern, SE-ARD kernel (RMSE = 0.02)']
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