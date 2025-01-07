# %%
import torch
import gpytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.lines import Line2D
import importlib

import pickle_reader
import analysis_utils
import gpt_utils

# Disable LaTeX renderingß to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
importlib.reload(analysis_utils)

# %%
# Load pickles from a directory
directory_path = '/Users/ikw/code/out_files/run2'
data_list = pickle_reader.collect_data_from_directory(directory_path, '.pickle')

# %%
df_original = pd.DataFrame(data_list)
# Display the DataFrame
print("DataFrame created from pickled dictionaries:")
df_original['env_desc'] = np.float64(df_original['env_desc'])
df_original['x_offset'] = np.float64(df_original['x_offset'])
df_original['y_offset'] = np.float64(df_original['y_offset'])
df_original['anisotropy'] = df_original['env_desc']
# Use angle_delta instead of rot
df_original['angle_delta'] = 90 - df_original['rot']

print(df_original)

# %%
# A lof of work to map c to aspect ratio
d = 0.01
 # Create environment x and y locations
scenario_x_min = 0
scenario_x_max = 250
scenario_y_min = 0
scenario_y_max = 250
sc_len_x = scenario_x_max - scenario_x_min
sc_len_y = scenario_y_max - scenario_y_min
sc_mean_x = scenario_x_min + sc_len_x/2.0
sc_mean_y = scenario_y_min + sc_len_y/2.0

env_x = np.linspace(scenario_x_min, scenario_x_max, sc_len_x, dtype=np.float64)
env_y = np.linspace(scenario_y_min, scenario_y_max, sc_len_y, dtype=np.float64)
env_xy = torch.tensor(np.column_stack((env_x, env_y)), dtype=torch.float64)
env_xy = gpytorch.utils.grid.create_data_from_grid(env_xy)

x_offset = 0
y_offset = 0
rel_dists = env_xy - torch.tensor([sc_mean_x - 25 + x_offset, sc_mean_y + y_offset])
distances = torch.norm(rel_dists, dim=1) # scale with dilution
angles = torch.atan2(rel_dists[:, 1], rel_dists[:, 0]) # [-pi, pi]

ratio = []
for c in np.arange(0, 200, 0.1):
    angles_cos = (-torch.cos(angles) + 1)*c
    angles_cos += 1
    env_values = gpt_utils.normalize_tensor(torch.exp(-(distances*d*(angles_cos))))
    mask = (env_values >= 0.40)
    height = env_xy[:, 1][mask].max() - env_xy[:, 1][mask].min()
    width = env_xy[:, 0][mask].max() - env_xy[:, 0][mask].min()
    ratio.append(width/height)

    #fig, ax = plt.subplots(figsize=(6, 6))
    #scatter = ax.scatter(env_xy[:, 0], env_xy[:, 1], c=env_values, cmap='coolwarm', s=2)
    #scatter = ax.scatter(env_xy[:, 0][mask], env_xy[:, 1][mask], color='black', s=1)
    #ax.set_title(f'ratio: {ratio[-1]:.3} ({width:.3}/{height:.3})')

    #plt.show()

indexer = analysis_utils.CustomIndexer(window_size=200, equal_left_right=True)

ratio_series = pd.Series(ratio)
ratio_series = ratio_series.rolling(indexer, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(np.arange(0, 200, 0.1), ratio, s=3)
ax.scatter(np.arange(0, 200, 0.1), ratio_series, s=1)
ax.set_title(f'Mapping c to plume aspect ratio', fontsize=16, fontweight='bold')
plt.show()

# %%
aspect_ratio = np.ndarray(len(df_original['anisotropy']), np.float32)
for i, c in enumerate(df_original['anisotropy'].values):
    aspect_ratio[i] = ratio_series.values[int((c-0.1)*10)]
    
df_original['anisotropy'] = aspect_ratio


# %%
# (TODO?): Adjust anisotropy to height/length ratio
# Show example simulations

# Create df_numerical and df_interesting
df_numerical = df_original.drop(['index', 'env_desc', 'env_num', 'spacing_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'kernel_type', 'grid_type'], axis=1)
df_interesting = df_original.drop(['index', 'env_desc', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'spacing_num', 'plume_samples'], axis=1)

# %%
# Create a heatmap, with only plain pattern entries
df = df_original.drop(['x_offset', 'y_offset', 'index', 'env_desc', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'spacing_num'], axis=1)
filter_values = {
        'grid_type': 'plain'
#        'kernel_type': 'SE-ARD'
}
df_filtered = analysis_utils.filter_df(df, filter_values)
df_corr = df_filtered.drop(['grid_type', 'kernel_type'], axis=1)

corr_matrix = df_corr.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

plt.figure(figsize=(12, 10))
#sns.heatmap(corr_matrix, annot=True, mask=mask, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0)
# Create the heatmap
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    mask=mask,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    linewidths=.5,
    vmin=-1.0,
    vmax=1.0,
)
plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.show()

# %%
# Pair plot
df_plain = df_filtered.drop(['grid_type'], axis=1)
sns.pairplot(df_plain, hue='kernel_type')
plt.show()

# %%
# Investigate the correlation between grid spacing and RMSE (and plume samples).
# Select all angles, only plain grid_type. Plot RMSE against spacing first for SE and then for SE-ARD.
# Define the constant values for other variables

spacings = {10, 20, 40}
kernel_types = ['SE', 'SE-ARD']

for i, kernel_type in enumerate(kernel_types):
    filter_values = {
        'angle_delta': [0, 90],
        'grid_type': 'plain',
        'kernel_type': kernel_type,
        'spacing': spacings
    }

    df_filtered = analysis_utils.filter_df(df_interesting, filter_values)
    # Sort the DataFrame
    df_filtered = df_filtered.sort_values(['spacing', 'anisotropy'])

    # Compute mean and standard deviation
    grouped = df_filtered.groupby(['anisotropy', 'spacing']).agg({'RMSE': ['mean', 'sem']}).reset_index()
    grouped.columns = ['anisotropy', 'spacing', 'RMSE_mean', 'RMSE_se']

    # Define the window size
    window_size = 20  # Adjust as needed

    # Sort the data
    grouped = grouped.sort_values(by=['spacing', 'anisotropy'])

    # Initialize a list to store the smoothed data
    smoothed_data = []

    # Apply rolling mean with adaptive window lengths
    for spacing in grouped['spacing'].unique():
        subset = grouped[grouped['spacing'] == spacing].copy()
        subset['RMSE_mean_smooth'] = subset['RMSE_mean'].rolling(
            window=window_size, min_periods=1, center=True).mean()
        subset['RMSE_se_smooth'] = subset['RMSE_se'].rolling(
            window=window_size, min_periods=1, center=True).mean()
        smoothed_data.append(subset)

    # Combine the smoothed data
    smoothed_grouped = pd.concat(smoothed_data)

    # Initialize the palette
    palette = sns.color_palette("Set2", n_colors=smoothed_grouped['spacing'].nunique())

    # Initialize the plot
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot the mean line for each 'spacing'
    for idx, spacing in enumerate(smoothed_grouped['spacing'].unique()):
        subset = smoothed_grouped[smoothed_grouped['spacing'] == spacing]
        ax.plot(
            subset['anisotropy'],
            subset['RMSE_mean_smooth'],
            label=f'{spacing} m',
            color=palette[idx],
            linewidth=2.0,
            markersize=8
        )
        
        # Plot dashed lines for upper and lower bounds
        ax.plot(
            subset['anisotropy'],
            subset['RMSE_mean_smooth'] + subset['RMSE_se_smooth'],
            linestyle='--',
            color=palette[idx],
            linewidth=1.0,
        )
        ax.plot(
            subset['anisotropy'],
            subset['RMSE_mean_smooth'] - subset['RMSE_se_smooth'],
            linestyle='--',
            color=palette[idx],
            linewidth=1.0,
        )

    # Adjust the labels and title
    ax.set_xlabel('Aspect ratio', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_title(f'RMSE vs. aspect ratio for different spacings ({kernel_type} kernel)', fontsize=16, fontweight='bold', x=(0.48-i/100.0))

    # Show the legend
    ax.legend(title='Spacing', fontsize=12, title_fontsize=13)

    # Display the plot
    plt.tight_layout()
    plt.savefig('figures/' + f'spacings_anisotropy_{kernel_type}.eps', format='eps', dpi=300)
    plt.show()

# %%
# Pattern type with unknown conditions - worst case
# Investigate effect of grid pattern vs. plain pattern (with the same number of samples).
# Compare difference in RMSE for all anisotropies

# Define merge keys
merge_keys = ['x_offset', 'y_offset', 'angle_delta', 'kernel_type', 'anisotropy']

# Plotting for Case 1
sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

plt.figure(figsize=(8, 6))
ax = plt.gca()

angle_deltas = [[0, 10], [40, 50], [80, 90]]
# Initialize the palette
palette = sns.color_palette("Set1", n_colors=len(angle_deltas))

for i, angle_d in enumerate(angle_deltas):
    # Case 1: Spacing 10 Plain vs. Spacing 20 Cross
    filter_values = {
        'angle_delta': angle_d,
        'grid_type': 'plain',
        'spacing': 10
    }

    config_A1 = analysis_utils.filter_df(df_interesting, filter_values)

    filter_values = {
        'angle_delta': angle_d,
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
    merged_df1['RMSE_difference'] = (merged_df1['RMSE_B'] / merged_df1['RMSE_A'])

    # Compute mean and standard deviation
    grouped = merged_df1.groupby(['anisotropy', 'kernel_type']).agg({'RMSE_difference': ['mean']}).reset_index()
    grouped.columns = ['anisotropy', 'kernel_type', 'RMSE_ratio_mean']

    # Sort the data
    grouped = grouped.sort_values(by=['kernel_type', 'anisotropy'])

    window_size = 40
    # Initialize a list to store the smoothed data
    smoothed_data = []

    # Apply rolling mean with adaptive window lengths
    for kernel in grouped['kernel_type'].unique():
        subset = grouped[grouped['kernel_type'] == kernel].copy()
        subset['RMSE_ratio_mean_smooth'] = subset['RMSE_ratio_mean'].rolling(
            window=window_size, min_periods=1, center=True).mean()
        smoothed_data.append(subset)

    # Combine the smoothed data
    smoothed_grouped = pd.concat(smoothed_data)

    # Plot the mean line for each 'spacing'
    for idx, kernel_type in enumerate(smoothed_grouped['kernel_type'].unique()):
        subset = smoothed_grouped[smoothed_grouped['kernel_type'] == kernel_type]
        ax.plot(
            subset['anisotropy'],
            subset['RMSE_ratio_mean_smooth'],
            label=(None, f'{angle_d}'+r'$^\circ$')[kernel_type == 'SE'],
            color=palette[i],
            linewidth=2.0,
            linestyle=('--', '-')[kernel_type == 'SE'],
            markersize=8
        )

plt.title('RMSE ratio vs. aspect ratio\n(Spacing 20 Grid / Spacing 10 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Aspect ratio', fontsize=14)
plt.ylabel('RMSE ratio', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title=r'$\delta$ angle', fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig('figures/' + f'grid_plain_20C-10P.eps', format='eps', dpi=300)
plt.show()


# %%
# Investigate effect of grid pattern vs. plain pattern (with the same number of samples).
# Compare difference in RMSE for all anisotropies

# Define merge keys
merge_keys = ['x_offset', 'y_offset', 'angle_delta', 'kernel_type', 'anisotropy']

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
merged_df1['RMSE_difference'] = (merged_df1['RMSE_B'] / merged_df1['RMSE_A'])

# Compute mean and standard deviation
grouped = merged_df1.groupby(['anisotropy', 'kernel_type']).agg({'RMSE_difference': ['mean']}).reset_index()
grouped.columns = ['anisotropy', 'kernel_type', 'RMSE_ratio_mean']

# Sort the data
grouped = grouped.sort_values(by=['kernel_type', 'anisotropy'])

window_size = 20
# Initialize a list to store the smoothed data
smoothed_data = []

# Apply rolling mean with adaptive window lengths
for kernel in grouped['kernel_type'].unique():
    subset = grouped[grouped['kernel_type'] == kernel].copy()
    subset['RMSE_ratio_mean_smooth'] = subset['RMSE_ratio_mean'].rolling(
        window=window_size, min_periods=1, center=True).mean()
    smoothed_data.append(subset)

# Combine the smoothed data
smoothed_grouped = pd.concat(smoothed_data)

# Plotting for Case 1
sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

# Initialize the palette
palette = sns.color_palette("Set1", n_colors=smoothed_grouped['kernel_type'].nunique())

plt.figure(figsize=(8, 6))
ax = plt.gca()

# Plot the mean line for each 'spacing'
for idx, kernel_type in enumerate(smoothed_grouped['kernel_type'].unique()):
    subset = smoothed_grouped[smoothed_grouped['kernel_type'] == kernel_type]
    ax.plot(
        subset['anisotropy'],
        subset['RMSE_ratio_mean_smooth'],
        label=kernel_type,
        color=palette[idx],
        linewidth=2.0,
        markersize=8
    )


""" sns.lineplot(
    data=merged_df1,
    x='anisotropy',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.0
) """
plt.title('RMSE ratio vs. Anisotropy\n(Spacing 20 Grid / Spacing 10 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Anisotropy', fontsize=14)
plt.ylabel('RMSE ratio', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel type', fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig('figures/' + f'grid_plain_20C-10P.eps', format='eps', dpi=300)
plt.show()

# Case 2: Spacing 20 Plain vs. Spacing 40 Cross
filter_values = {
    'angle_delta': [0, 10],
    'grid_type': 'plain',
    'spacing': 20
}

config_A2 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'angle_delta': [0, 10],
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
merged_df2['RMSE_difference'] = (merged_df2['RMSE_B'] / merged_df2['RMSE_A'])

# Compute mean and standard deviation
grouped = merged_df2.groupby(['anisotropy', 'kernel_type']).agg({'RMSE_difference': ['mean']}).reset_index()
grouped.columns = ['anisotropy', 'kernel_type', 'RMSE_ratio_mean']

# Sort the data
grouped = grouped.sort_values(by=['kernel_type', 'anisotropy'])

# Initialize a list to store the smoothed data
smoothed_data = []

# Apply rolling mean with adaptive window lengths
for kernel in grouped['kernel_type'].unique():
    subset = grouped[grouped['kernel_type'] == kernel].copy()
    subset['RMSE_ratio_mean_smooth'] = subset['RMSE_ratio_mean'].rolling(
        window=window_size, min_periods=1, center=True).mean()
    smoothed_data.append(subset)

# Combine the smoothed data
smoothed_grouped = pd.concat(smoothed_data)

# Plotting for Case 1
sns.set_theme(font='Dejavu Serif', style='whitegrid', context='paper')

plt.figure(figsize=(8, 6))
ax = plt.gca()

# Plot the mean line for each 'spacing'
for idx, kernel_type in enumerate(smoothed_grouped['kernel_type'].unique()):
    subset = smoothed_grouped[smoothed_grouped['kernel_type'] == kernel_type]
    ax.plot(
        subset['anisotropy'],
        subset['RMSE_ratio_mean_smooth'],
        label=kernel_type,
        color=palette[idx],
        linewidth=2.0,
        markersize=8
    )

plt.title('RMSE ratio vs. Anisotropy\n(Spacing 40 Grid / Spacing 20 Plain)', fontsize=16, fontweight='bold')
plt.xlabel('Anisotropy', fontsize=14)
plt.ylabel('RMSE ratio', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel type', fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig('figures/' + f'grid_plain_40C-20P.eps', format='eps', dpi=300)
plt.show()

# %%
# Angle delta
plt.figure(figsize=(8, 6))
ax = plt.gca()

kernels = ['SE', 'SE-ARD']
spaces = [10, 20, 40]

# Initialize the palette
palette = sns.color_palette("Set2", n_colors=len(spaces))

# Compute mean and standard deviation
filter1 = {
    'grid_type': 'plain'
}

df_filter1 = analysis_utils.filter_df(df_interesting, filter1)
grouped = df_filter1.groupby(['angle_delta', 'kernel_type', 'spacing']).agg({'RMSE': ['mean']}).reset_index()
grouped.columns = ['angle_delta', 'kernel_type', 'spacing', 'RMSE_mean']

# Sort the data
grouped = grouped.sort_values(by=['spacing', 'kernel_type', 'angle_delta'])

# Initialize the palette
#palette = sns.color_palette("Set2", n_colors=smoothed_grouped['spacing'].nunique())

for i, kernel in enumerate(kernels):
    for j, space in enumerate(spaces):
        filters = {
            'spacing': space,
            'kernel_type': kernel
        }

        # Filter the DataFrame
        df_filtered = analysis_utils.filter_df(grouped, filters)
        # Ensure data is sorted by 'anisotropy'
        df_filtered = df_filtered.sort_values('angle_delta')

        ax.plot(
            df_filtered['angle_delta'],
            df_filtered['RMSE_mean'],
            color=palette[j],
            linewidth=2.0,
            linestyle=('--', '-')[kernel == 'SE'],
            markersize=8
        )

# Create custom legend handles for spacings (colors)
spacing_handles = [Line2D([0], [0], color=palette[j], lw=2) for j in range(len(spaces))]
spacing_labels = [f'{space} m' for space in spaces]

# Create custom legend handles for kernels (line styles)
kernel_linestyles = ['-', '--']
kernel_handles = [Line2D([0], [0], color='black', lw=2, linestyle=style) for style in kernel_linestyles]
kernel_labels = ['SE', 'SE-ARD']

# Add the legends to the plot
legend1 = ax.legend(handles=spacing_handles, labels=spacing_labels, title='Spacing', fontsize=12, title_fontsize=13)
ax.add_artist(legend1)  # Add the first legend manually
#legend2 = ax.legend(handles=kernel_handles, labels=kernel_labels, title='Kernel Type', loc='upper left', fontsize=12, title_fontsize=13)
#r'$\alpha$=' + f"{advection_angles[i]}

plt.title('RMSE vs. relative orientation ' + r'$\delta$', fontsize=16, fontweight='bold')
plt.xlabel(r'$\delta$' + r' [$\circ$]', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
#plt.legend(title='Spacing, Kernel type', fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig('figures/' + f'RMSE_angle_delta.eps', format='eps', dpi=300)
plt.show()



# %%
# Investigate effect of angle_delta

angle_delta_intervals = {
    '0-30': [0, 30],
    '31-60': [31, 60],
    '61-90': [61, 90]
}
angle_delta_intervals_narrow = {
    '0-10': [0, 10],
    '11-20': [11, 20],
    '21-30': [21, 30],
    '31-40': [31, 40],
    '41-50': [41, 50],
    '51-60': [51, 60],
    '61-70': [61, 70],
    '71-80': [71, 80],
    '81-90': [81, 90]
}

for kernel in kernel_types:
    plt.figure(figsize=(8, 6))
    
    # Initialize a color palette
    palette = sns.color_palette("tab10", n_colors=len(angle_delta_intervals_narrow))
    
    # Loop over each rot interval
    for idx, (interval_name, interval_range) in enumerate(angle_delta_intervals_narrow.items()):
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
    'angle_delta': [80, 90],
    'spacing': {10, 20, 40}
}

# Filter the DataFrame
df = analysis_utils.filter_df(df, filters)

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='RMSE', hue='spacing', kde=False, multiple='stack')
plt.title('Distribution of RMSE', fontsize=16, fontweight='bold')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.savefig('figures/' + f'env_histogram.eps', format='eps', dpi=300)
plt.show()

# Create the figure and axes
fig, ax = plt.subplots(figsize=(6, 4))
# Plot the boxplot
sns.boxplot(x='kernel_type', y='RMSE', hue='spacing', data=df, whis=(0, 100), ax=ax)
# Set the main title using suptitle
fig.suptitle('RMSE by Kernel Type and Line Spacing', fontsize=16, fontweight='bold')
# Set the subtitle using ax.set_title
ax.set_title(r'$\delta \in [80, 90]^\circ$', fontsize=12)
# Adjust layout to make space for titles
plt.tight_layout(rect=[0, 0, 1, 1.03])
# Set labels and legend
ax.set_xlabel('Kernel Type')
ax.set_ylabel('RMSE')
ax.legend(title='Spacing')
# Save and display the plot
plt.savefig('figures/env_boxplot.eps', format='eps', dpi=300)
plt.show()


 # %%
# netCDF analysis
 # Load a single .pickle
#file_path = '/Users/ikw/code/out_files/c_scale_test/2024-11-20 13_00_25_84_output.pickle'
#data_list = pickle_reader.load_pickled_dicts_from_file(file_path)

# %%
# netCDF analysis - Load pickles from a directory
directory_path = '/Users/ikw/code/out_files/netCDF_run2'
data_list = pickle_reader.collect_data_from_directory(directory_path, '.pickle')
environments = pickle_reader.collect_data_from_directory(directory_path + '/figures/environments', '.pickle')
advection_angles = [5, -70, -50, -20]

# %%
df_original = pd.DataFrame(data_list)
# Set up df angle_delta

df_original['advection_angle'] = [advection_angles[int(a)] for a in df_original['env_num'].values]
df_original['angle_delta'] = df_original['rot'] + 90 - df_original['advection_angle']

# Compute ts, depth
df_original['file'] = [file.split('/')[-1].split(' ')[0] for file in df_original['env_desc']]
df_original['ts'] = [(int(file.split('/')[-1].split(' ')[0].split('-')[-1].split('.')[0]) - 1)*12 + int(file.split(' ')[-3]) for file in df_original['env_desc']]
df_original['depth'] = [int(file.split(' ')[-1]) for file in df_original['env_desc']]

# Create df_numerical and df_interesting
df_numerical = df_original.drop(['index', 'env_desc', 'env_num', 'spacing_num', 'rot_num', 'ct_train', 'ct_predict', 'kernel_type', 'grid_type'], axis=1)
df_interesting = df_original.drop(['index', 'env_desc', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'spacing_num', 'x_offset', 'y_offset', 'file'], axis=1)
print(df_interesting)

# %%
# Assuming 'environments' is your list of figures

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

fig_combined.suptitle('Emission snapshots', fontsize=16, fontweight='bold', x=0.4, y=0.96)

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
spacings = [10, 20, 40]
kernel_types = ['SE', 'SE-ARD']

# Filter values (include both kernel_types by not specifying 'kernel_type' in the filter)
filter_values = {
    'angle_delta': [80, 100],
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
g.figure.subplots_adjust(top=0.84)

# Adjust the spacing between subplots
g.figure.subplots_adjust(hspace=0.2, wspace=0.1)

# Rotate x-axis labels if needed to fit spacing labels
for ax in g.axes.flatten():
    ax.tick_params(axis='x')

# Manually set the titles for each subplot

for i, ax in enumerate(g.axes.flat):
    ax.set_title(f"{df_original['ts'].unique()[i]*10} minutes, " + r'$\alpha$=' + f"{advection_angles[i]}")

# Add a suptitle
g.figure.suptitle('RMSE by snapshot, spacing, and kernel type', fontsize=16, fontweight='bold')
# Add the subtitle beneath the main title
g.figure.text(
    0.5, 0.91,  # Adjust the y-coordinate as needed
    r'$\delta \in [80, 100]^\circ$',
    ha='center',
    fontsize=12
)

# Adjust the layout to make space for the titles
g.figure.subplots_adjust(top=0.86)  # Adjust top to make room for the suptitle and subtitle


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
