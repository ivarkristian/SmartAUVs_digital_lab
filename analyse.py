# %%
import pandas as pd
import pickle_reader
 
 # %%
 # Load a single .pickle

file_path = '/Users/ikw/code/out_files/2024-11-14 10_56_42_84_output.pickle'
data_list = pickle_reader.load_pickled_dicts_from_file(file_path)
df = pd.DataFrame(data_list)

# Display the DataFrame
print("DataFrame created from pickled dictionaries:")
print(df)

# %%
