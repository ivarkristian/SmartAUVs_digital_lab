# %%
import pandas as pd
import pickle_reader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
 # %%
 # Load a single .pickle

file_path = '/Users/ikw/code/out_files/2024-11-14 10_56_42_84_output.pickle'
data_list = pickle_reader.load_pickled_dicts_from_file(file_path)
df = pd.DataFrame(data_list)

# Display the DataFrame
print("DataFrame created from pickled dictionaries:")
df['env_desc'] = np.float64(df['env_desc'])
df['y_offset'] = np.float64(df['y_offset'])
print(df)

# %%
# Create df_numerical

df_numerical = df.drop(['index', 'env_num', 'spacing_num', 'rot_num', 'ct_train', 'ct_predict', 'kernel_type', 'grid_type'], axis=1)
df_interesting = df.drop(['index', 'env_num', 'rot_num', 'ct_train', 'ct_predict', 'rot', 'spacing_num'], axis=1)


# %%
corr_matrix = df_numerical.corr()

# %%
# Set the style of the seaborn plot
sns.set_theme(style='whitegrid', context='notebook')

# Create the pair plot
pair_plot = sns.pairplot(df_interesting, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, diag_kws={'fill': True})

# Add a title to the plot
pair_plot.figure.suptitle('Pairwise Scatter Plots', fontsize=16)
pair_plot.figure.subplots_adjust(top=0.95)  # Adjust subplots to fit the title

plt.show()

# %%
# 0. Confirm the obvious: Denser grid gives lower RMSE and more plume samples. Keep angle constant, compare RMSE with spacing first for SE and then for SE-ARD.

# 1. Which angle should we fly at? How important is the angle for different wind speeds? Keep kernel and spacing constant, check RMSE for all angles, at three different wind speeds, first for SE then for SE-ARD, six cases in total.

# 2. Is cross-pattern better than decreasing spacing? Compare for each spacing. Check RMSE for three different angles and both kernels, six cases in total.

# 3. 

