# %%
import pandas as pd
import pickle_reader
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg
import seaborn as sns
import analysis_utils
import matplotlib.cm as cm
import matplotlib.colors as colors

# Disable LaTeX renderingß to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})
# %%
# Load pickles from a directory
directory_path = '/Users/ikw/code/out_files/run1'
data_list = pickle_reader.collect_data_from_directory(directory_path, '.pickle')

# %%
df_original = pd.DataFrame(data_list)
# Display the DataFrame
print("DataFrame created from pickled dictionaries:")
df_original['env_desc'] = np.float64(df_original['env_desc'])
#df_original['x_offset'] = np.float64(df_original['x_offset'])
df_original['y_offset'] = np.float64(df_original['y_offset'])
df_original['anisotropy'] = df_original['env_desc']
df_original['angle_delta'] = 90 - df_original['rot']

print(df_original)

# %%
# (TODO?): Adjust anisotropy to height/length ratio
# Show example simulations

# Use angle_delta instead of rot
# angle_delta = anisotropy_angle - rot_dir, and anisotropy_angle was set to 0 for elliptical environments.
# We want a grid angle that describes the angle between flight lines and anisotropy. Since rot_dir = 0 corresponds to a grid angle
# of 90, and we simulate rot_dir 0-90, we can compute the grid angle as 90-rot_dir  (Or angle_delta + 90).

# Create df_numerical and df_interesting
df_numerical = df_original.drop(['index', 'env_desc', 'env_num', 'spacing_num', 'rot_num', 'ct_train', 'ct_predict', 'kernel_type', 'grid_type'], axis=1)
df_interesting = df_original.drop(['index', 'env_desc', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'spacing_num', 'plume_samples'], axis=1)

# %%
corr_matrix = df_numerical.corr()


# %%
# Investigate the correlation between grid spacing and RMSE (and plume samples).
# Select all angles, only plain grid_type. Plot RMSE against spacing first for SE and then for SE-ARD.
# Define the constant values for other variables

spacings = [10, 20, 30, 40]
kernel_types = ['SE', 'SE-ARD']

for i, kernel_type in enumerate(kernel_types):
    filter_values = {
        'angle_delta': [0, 90],
        'grid_type': 'plain',
        'kernel_type': kernel_type
    }

    df_filtered = analysis_utils.filter_df(df_interesting, filter_values)
    # Sort the DataFrame
    df_filtered = df_filtered.sort_values(['spacing', 'anisotropy'])

    # Get unique spacing values
    spacing_values = sorted(df_filtered['spacing'].unique())

    # Set the plotting style and palette
    sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

    # Create a custom color palette
    palette = sns.color_palette("viridis", n_colors=len(spacing_values))

    # Create the line plot
    plt.figure(figsize=(8, 6))

    sns.lineplot(
        data=df_filtered,
        x='anisotropy',
        y='RMSE',
        hue='spacing',
        palette=palette,
        linewidth=2.0,
        #marker='o',
        markersize=8
    )

    # Customize the plot
    plt.title(f'RMSE vs. anisotropy for different spacings ({kernel_type} kernel)', fontsize=16, fontweight='bold', x=(0.48-i/100.0))
    plt.xlabel('Anisotropy', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(title='Spacing', fontsize=12, title_fontsize=13)
    plt.tight_layout()

    plt.savefig('figures/' + f'fig1_{kernel_type}.eps', format='eps', dpi=300)
    # Show the plot
    plt.show()

# %%
# !!!Remember x_offset below!!! Investigate effect of grid pattern vs. plain pattern (with the same number of samples).
# Compare difference in RMSE for all anisotropies

# Define merge keys
merge_keys = ['y_offset', 'angle_delta', 'kernel_type', 'anisotropy']

# Case 1: Spacing 10 Plain vs. Spacing 20 Cross
filter_values = {
    'angle_delta': [0, 90],
    'grid_type': 'plain',
    'spacing': 10
}

config_A1 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'angle_delta': [0, 90],
    'grid_type': 'cross',
    'spacing': 20
}

config_B1 = analysis_utils.filter_df(df_interesting, filter_values)

merged_df1 = pd.merge(
    config_A1,
    config_B1,
    on=merge_keys,
    suffixes=('_A', '_B')
)

# Compute RMSE difference
merged_df1['RMSE_difference'] = merged_df1['RMSE_B'] - merged_df1['RMSE_A']

# Sort data
merged_df1 = merged_df1.sort_values('anisotropy')

# Plotting for Case 1
sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

plt.figure(figsize=(8, 6))
sns.lineplot(
    data=merged_df1,
    x='anisotropy',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.0
)
plt.title('Difference in RMSE vs. Anisotropy\n(Spacing 20 Cross - Spacing 10 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Anisotropy', fontsize=14)
plt.ylabel('RMSE Difference', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel Type')
plt.tight_layout()
plt.savefig('figures/' + f'fig2_20C-10P.eps', format='eps', dpi=300)
plt.show()

# Case 2: Spacing 20 Plain vs. Spacing 40 Cross
filter_values = {
    'angle_delta': [0, 90],
    'grid_type': 'plain',
    'spacing': 20
}

config_A2 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'angle_delta': [0, 90],
    'grid_type': 'cross',
    'spacing': 40
}

config_B2 = analysis_utils.filter_df(df_interesting, filter_values)

merged_df2 = pd.merge(
    config_A2,
    config_B2,
    on=merge_keys,
    suffixes=('_A', '_B')
)

# Compute RMSE difference
merged_df2['RMSE_difference'] = merged_df2['RMSE_B'] - merged_df2['RMSE_A']

# Sort data
merged_df2 = merged_df2.sort_values('anisotropy')

# Plotting for Case 2
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=merged_df2,
    x='anisotropy',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.5,
    palette='Set2'
)
plt.title('Difference in RMSE vs. Anisotropy\n(Spacing 40 Cross - Spacing 20 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Anisotropy', fontsize=14)
plt.ylabel('RMSE Difference', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel Type')
plt.tight_layout()
plt.savefig('figures/' + f'fig2_40C-20P.eps', format='eps', dpi=300)
plt.show()

# %%
# Investigate effect of angle_delta

angle_delta_intervals = {
    '0-30': [0, 30],
    '31-60': [31, 60],
    '61-90': [61, 90]
}

for kernel in kernel_types:
    plt.figure(figsize=(8, 6))
    
    # Initialize a color palette
    palette = sns.color_palette("tab10", n_colors=len(angle_delta_intervals))
    
    # Loop over each rot interval
    for idx, (interval_name, interval_range) in enumerate(angle_delta_intervals.items()):
        # Define filters
        filters = {
            'grid_type': 'plain',
            'kernel_type': kernel,
            'angle_delta': interval_range  # Use the interval for 'rot'
        }
        
        # Filter the DataFrame
        df_filtered = analysis_utils.filter_df(df_interesting, filters)
        
        # Ensure data is sorted by 'anisotropy'
        df_filtered = df_filtered.sort_values('anisotropy')
        
        # Check if the filtered DataFrame is not empty
        if not df_filtered.empty:
            # Plot RMSE vs anisotropy
            sns.lineplot(
                data=df_filtered,
                x='anisotropy',
                y='RMSE',
                label=f'{interval_name}',
                color=palette[idx],
                linewidth=2.5,
                markersize=8
            )
        else:
            print(f"No data for kernel_type '{kernel}' and angle_delta interval '{interval_name}'.")
    
    # Customize the plot
    plt.title(f'RMSE vs. anisotropy for kernel type: {kernel}', fontsize=16, fontweight='bold')
    plt.xlabel('Anisotropy', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(title='angle_delta intervals', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig('figures/' + f'fig3_{kernel}.eps', format='eps', dpi=300)
    
    # Show the plot
    plt.show()


# %%
# Statistical significance analysis

# %%
# # Load and clean data

# Assuming 'df_interesting' is your DataFrame
df = df_interesting.copy()

# Ensure correct data types
df['anisotropy'] = pd.to_numeric(df['anisotropy'], errors='coerce')
df['angle_delta'] = pd.to_numeric(df['angle_delta'], errors='coerce')
df['y_offset'] = pd.to_numeric(df['y_offset'], errors='coerce')
df['spacing'] = pd.to_numeric(df['spacing'], errors='coerce')
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')

# Convert categorical variables
df['kernel_type'] = df['kernel_type'].astype('category')
df['grid_type'] = df['grid_type'].astype('category')

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values in critical columns
df.dropna(subset=['anisotropy', 'angle_delta', 'kernel_type', 'grid_type', 'RMSE'], inplace=True)

# %%
# Exploratory analysis using 80-90 degrees angle_delta
filters = {
    'grid_type': 'plain',
    'angle_delta': [80, 90]
}

# Filter the DataFrame
df = analysis_utils.filter_df(df, filters)

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='RMSE', hue='spacing', kde=False, multiple='stack')
#sns.histplot(df['RMSE'], kde=True)
plt.title('Distribution of RMSE', fontsize=16, fontweight='bold')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.savefig('figures/' + f'fig4_histogram.eps', format='eps', dpi=300)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='kernel_type', y='RMSE', hue='spacing', data=df)
plt.title('RMSE by Kernel Type and grid spacing', fontsize=16, fontweight='bold')
plt.xlabel('Kernel Type')
plt.ylabel('RMSE')
plt.legend(title='Spacing')
plt.tight_layout()
plt.savefig('figures/' + f'fig4_boxplot.eps', format='eps', dpi=300)
plt.show()


 # %%
# netCDF analysis
 # Load a single .pickle
#file_path = '/Users/ikw/code/out_files/c_scale_test/2024-11-20 13_00_25_84_output.pickle'
#data_list = pickle_reader.load_pickled_dicts_from_file(file_path)

# %%
# netCDF analysis - Load pickles from a directory
directory_path = '/Users/ikw/code/out_files/netCDF_run1'
data_list = pickle_reader.collect_data_from_directory(directory_path, '.pickle')
environments = pickle_reader.collect_data_from_directory(directory_path + '/figures/environments', '.pickle')

# %%
df_original = pd.DataFrame(data_list)
# Set up df angle_delta
df_original['angle_delta'] = 90 - df_original['rot']

# Compute ts, depth
df_original['file'] = [file.split('/')[-1].split(' ')[0] for file in df_original['env_desc']]
df_original['ts'] = [(int(file.split('/')[-1].split(' ')[0].split('-')[-1].split('.')[0]) - 1)*12 + int(file.split(' ')[-3]) for file in df_original['env_desc']]
df_original['depth'] = [int(file.split(' ')[-1]) for file in df_original['env_desc']]

# Create df_numerical and df_interesting
df_numerical = df_original.drop(['index', 'env_desc', 'env_num', 'spacing_num', 'rot_num', 'ct_train', 'ct_predict', 'kernel_type', 'grid_type'], axis=1)
df_interesting = df_original.drop(['index', 'env_desc', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'spacing_num', 'y_offset', 'file'], axis=1)
print(df_interesting)

# %%
# Assuming 'environments' is your list of figures
advection_angles = [5, -75, -50, -20]
tss = df_original['ts'].unique()
titles = []
for i, angle in enumerate(advection_angles):
    titles.append(str(tss[i]*10) + ' minutes,' + r" $\alpha$ = " + str(angle))

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
fig_combined, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# Display images in subplots
for i, (img, ax) in enumerate(zip(image_list, axes)):
    ax.imshow(img)
    ax.axis('off')
    #ax.set_title(titles[i], fontsize=12)

fig_combined.suptitle('Emission scenarios', fontsize=16, fontweight='bold', x=0.4, y=0.96)

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
fig_combined.savefig('figures/' + 'netCDF_scenarios.eps', format='eps', dpi=300)

# Display the combined figure
plt.show()

# %%
# Investigate the correlation between grid spacing and RMSE (and plume samples).
# Select all angles, only plain grid_type. Plot RMSE against spacing first for SE and then for SE-ARD.
# Define the constant values for other variables

# Define spacings and kernel_types
spacings = [10, 20, 30, 40]
kernel_types = ['SE', 'SE-ARD']

# Filter values (include both kernel_types by not specifying 'kernel_type' in the filter)
filter_values = {
    'angle_delta': [0, 90],
    'grid_type': 'plain'
}

# Filter the DataFrame using your filter_df function
df_filtered = analysis_utils.filter_df(df_interesting, filter_values)

# Filter to include only the desired kernel_types, in case there are more
df_filtered = df_filtered[df_filtered['kernel_type'].isin(kernel_types)]

# Convert 'ts' and 'spacing' to categorical variables with a specific order
df_filtered['ts'] = df_filtered['ts'].astype('category')
df_filtered['spacing'] = pd.Categorical(df_filtered['spacing'], categories=spacings, ordered=True)

# Sort the DataFrame
df_filtered = df_filtered.sort_values(['ts', 'spacing'])

# Set the plotting style and palette
sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

# Create a custom color palette for kernel_types
palette = sns.color_palette("Set2", n_colors=len(kernel_types))

# Create a FacetGrid with adjusted height and aspect ratio
g = sns.FacetGrid(df_filtered, col='ts', col_wrap=2, height=3, aspect=1, sharey=True)

# Map a bar plot to each subplot with adjusted bar width
g.map_dataframe(
    sns.barplot,
    x='spacing',
    y='RMSE',
    hue='kernel_type',
    hue_order=kernel_types,
    palette=palette,
    dodge=True,
    width=0.6,  # Adjust bar width as needed
    errorbar=None
)

# Adjust the titles and labels
g.set_titles('Time step: {col_name}')
g.set_axis_labels('Spacing', 'RMSE')

# Add a legend for kernel_type
g.add_legend(title='Kernel Type', fontsize=10, title_fontsize=11)

# Adjust layout to reduce space between suptitle and subplots
g.figure.subplots_adjust(top=0.88)

# Adjust the spacing between subplots
g.figure.subplots_adjust(hspace=0.2, wspace=0.1)

# Rotate x-axis labels if needed to fit spacing labels
for ax in g.axes.flatten():
    ax.tick_params(axis='x')

# Add a suptitle
g.figure.suptitle('RMSE by time step, spacing, and kernel type', fontsize=16, fontweight='bold')
g.savefig('figures/' + 'netCDF_spacings.eps', format='eps', dpi=300)
# Show the plot
plt.show()

# %%
# !!!Remember x_offset below!!! Investigate effect of grid pattern vs. plain pattern (with the same number of samples).
# Compare difference in RMSE for all anisotropies

# Define merge keys
merge_keys = ['angle_delta', 'kernel_type']

# Case 1: Spacing 10 Plain vs. Spacing 20 Cross
filter_values = {
    'angle_delta': [0, 180],
    'grid_type': 'plain',
    'spacing': 10
}

config_A1 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'angle_delta': [0, 180],
    'grid_type': 'cross',
    'spacing': 20
}

config_B1 = analysis_utils.filter_df(df_interesting, filter_values)

merged_df1 = pd.merge(
    config_A1,
    config_B1,
    on=merge_keys,
    suffixes=('_A', '_B')
)

# Compute RMSE difference
merged_df1['RMSE_difference'] = merged_df1['RMSE_B'] - merged_df1['RMSE_A']

# Sort data
merged_df1 = merged_df1.sort_values('angle_delta')

# Plotting for Case 1
sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

plt.figure(figsize=(8, 4))
sns.lineplot(
    data=merged_df1,
    x='angle_delta',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.5
)
plt.title('Difference in RMSE vs. angle_delta\n(Spacing 20 Cross - Spacing 10 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Angle_delta', fontsize=14)
plt.ylabel('RMSE Difference', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel Type')
plt.tight_layout()
plt.savefig('figures/' + 'netCDF_20C-10P.eps', format='eps', dpi=300)
plt.show()

# Case 2: Spacing 20 Plain vs. Spacing 40 Cross
filter_values = {
    'angle_delta': [0, 180],
    'grid_type': 'plain',
    'spacing': 20
}

config_A2 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'angle_delta': [0, 180],
    'grid_type': 'cross',
    'spacing': 40
}

config_B2 = analysis_utils.filter_df(df_interesting, filter_values)

merged_df2 = pd.merge(
    config_A2,
    config_B2,
    on=merge_keys,
    suffixes=('_A', '_B')
)

# Compute RMSE difference
merged_df2['RMSE_difference'] = merged_df2['RMSE_B'] - merged_df2['RMSE_A']

# Sort data
merged_df2 = merged_df2.sort_values('angle_delta')

# Plotting for Case 2
plt.figure(figsize=(8, 4))
sns.lineplot(
    data=merged_df2,
    x='angle_delta',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.5,
    palette='Set2'
)
plt.title('Difference in RMSE vs. angle_delta\n(Spacing 40 Cross - Spacing 20 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Angle_delta', fontsize=14)
plt.ylabel('RMSE Difference', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel Type')
plt.tight_layout()
plt.savefig('figures/' + 'netCDF_40C-20P.eps', format='eps', dpi=300)
plt.show()

# %%
