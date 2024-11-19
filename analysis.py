# %%
import pandas as pd
import pickle_reader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_utils
 
 # %%
 # Load a single .pickle
file_path = '/Users/ikw/code/out_files/2024-11-14 10_56_42_84_output.pickle'
data_list = pickle_reader.load_pickled_dicts_from_file(file_path)

# %%
# Load pickles from a directory
directory_path = '/Users/ikw/code/out_files/run1'
data_list = pickle_reader.collect_data_from_directory(directory_path, '.pickle')

# %%
df_original = pd.DataFrame(data_list)
# Display the DataFrame
print("DataFrame created from pickled dictionaries:")
df_original['env_desc'] = np.float64(df_original['env_desc'])
df_original['y_offset'] = np.float64(df_original['y_offset'])
print(df_original)

# %%
# Make adjustments to df_original
df_original['anisotropy'] = df_original['env_desc']

# TODO:
# Adjust anisotropy to height/length ratio

# Use angle_delta instead of rot

# Create df_numerical and df_interesting
df_numerical = df_original.drop(['index', 'env_desc', 'env_num', 'spacing_num', 'rot_num', 'ct_train', 'ct_predict', 'kernel_type', 'grid_type'], axis=1)
df_interesting = df_original.drop(['index', 'env_desc', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'angle_delta', 'spacing_num', 'plume_samples'], axis=1)

# %%
corr_matrix = df_numerical.corr()



# %%
# Investigate the correlation between grid spacing and RMSE (and plume samples).
# Select all angles, only plain grid_type. Plot RMSE against spacing first for SE and then for SE-ARD.
# Define the constant values for other variables

spacings = [10, 20, 30, 40]
kernel_types = ['SE', 'SE-ARD']

for kernel_type in kernel_types:
    filter_values = {
        'rot': [0, 90],
        'grid_type': 'plain',
        'kernel_type': kernel_type
    }

    df_filtered = analysis_utils.filter_df(df_interesting, filter_values)
    # Sort the DataFrame
    df_filtered = df_filtered.sort_values(['spacing', 'anisotropy'])

    # Get unique spacing values
    spacing_values = sorted(df_filtered['spacing'].unique())

    # Set the plotting style and palette
    sns.set_theme(style='whitegrid', context='paper')

    # Create a custom color palette
    palette = sns.color_palette("viridis", n_colors=len(spacing_values))

    # Create the line plot
    plt.figure(figsize=(12, 8))

    sns.lineplot(
        data=df_filtered,
        x='anisotropy',
        y='RMSE',
        hue='spacing',
        palette=palette,
        linewidth=2.5,
        #marker='o',
        markersize=8
    )

    # Customize the plot
    plt.title(f'RMSE vs. Anisotropy for different grid spacings (Kernel: {kernel_type})', fontsize=16)
    plt.xlabel('Anisotropy', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(title='Spacing', fontsize=12, title_fontsize=13)
    plt.tight_layout()

    # Show the plot
    plt.show()

# %%
# !!!Remember x_offset below!!! Investigate effect of grid pattern vs. plain pattern (with the same number of samples).
# Compare difference in RMSE for all anisotropies

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'anisotropy' is numeric
#df_interesting['anisotropy'] = pd.to_numeric(df_interesting['anisotropy'], errors='coerce')

# Define merge keys
merge_keys = ['y_offset', 'rot', 'kernel_type', 'anisotropy']

# Case 1: Spacing 10 Plain vs. Spacing 20 Cross
filter_values = {
    'rot': [0, 90],
    'grid_type': 'plain',
    'spacing': 10
}

config_A1 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'rot': [0, 90],
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
sns.set_theme(style='whitegrid', context='paper')

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=merged_df1,
    x='anisotropy',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.5
)
plt.title('Difference in RMSE vs. Anisotropy\n(Spacing 20 Cross - Spacing 10 Plain)', fontsize=16)
plt.xlabel('Anisotropy', fontsize=14)
plt.ylabel('RMSE Difference', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel Type')
plt.tight_layout()
plt.show()

# Case 2: Spacing 20 Plain vs. Spacing 40 Cross
filter_values = {
    'rot': [0, 90],
    'grid_type': 'plain',
    'spacing': 20
}

config_A2 = analysis_utils.filter_df(df_interesting, filter_values)

filter_values = {
    'rot': [0, 90],
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
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=merged_df2,
    x='anisotropy',
    y='RMSE_difference',
    hue='kernel_type',  # If multiple kernel types
    linewidth=2.5,
    palette='Set2'
)
plt.title('Difference in RMSE vs. Anisotropy\n(Spacing 40 Cross - Spacing 20 Plain)', fontsize=16)
plt.xlabel('Anisotropy', fontsize=14)
plt.ylabel('RMSE Difference', fontsize=14)
#plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Kernel Type')
plt.tight_layout()
plt.show()

# %%
# 
rot_intervals = {
    '0-30': [0, 30],
    '31-60': [31, 60],
    '61-90': [61, 90]
}

for kernel in kernel_types:
    plt.figure(figsize=(12, 8))
    
    # Initialize a color palette
    palette = sns.color_palette("tab10", n_colors=len(rot_intervals))
    
    # Loop over each rot interval
    for idx, (interval_name, interval_range) in enumerate(rot_intervals.items()):
        # Define filters
        filters = {
            'grid_type': 'plain',
            'kernel_type': kernel,
            'rot': interval_range  # Use the interval for 'rot'
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
                label=f'rot {interval_name}',
                color=palette[idx],
                linewidth=2.5,
                markersize=8
            )
        else:
            print(f"No data for kernel_type '{kernel}' and rot interval '{interval_name}'.")
    
    # Customize the plot
    plt.title(f'RMSE vs. Anisotropy for Kernel Type: {kernel}', fontsize=16)
    plt.xlabel('Anisotropy', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(title='Rotation Intervals', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    
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
df['rot'] = pd.to_numeric(df['rot'], errors='coerce')
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
df.dropna(subset=['anisotropy', 'rot', 'kernel_type', 'grid_type', 'RMSE'], inplace=True)

# %%
# Exploratory analysis

plt.figure(figsize=(10, 6))
sns.histplot(df['RMSE'], kde=True)
plt.title('Distribution of RMSE')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='kernel_type', y='RMSE', hue='grid_type', data=df)
plt.title('RMSE by Kernel Type and Grid Type')
plt.xlabel('Kernel Type')
plt.ylabel('RMSE')
plt.legend(title='Grid Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='rot', y='RMSE', hue='kernel_type', style='grid_type', data=df)
plt.title('RMSE vs. Rot')
plt.xlabel('Rotation (rot)')
plt.ylabel('RMSE')
plt.legend(title='Kernel/Grid Type')
plt.show()





# %%
# Compare using rot <30 deg

# 1. Which angle should we fly at? How important is the angle for different wind speeds? Keep kernel and spacing constant, check RMSE for all angles, at three different wind speeds, first for SE then for SE-ARD, six cases in total.

# 2. Is cross-pattern better than decreasing spacing? Compare for each spacing. Check RMSE for three different angles and both kernels, six cases in total.

# 3. 



# %%
