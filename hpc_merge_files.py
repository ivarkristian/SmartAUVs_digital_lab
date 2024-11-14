# %%
import pickle_reader
import pandas as pd
import argparse
from datetime import datetime
import os

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ftype", type=str, help="File postfix to parse, e.g. \'.pickle\'")
parser.add_argument("--path", type=str, help="Path to file directory, e.g. my_files")
args = parser.parse_args()
#ftype = args.ftype
ftype = '.pickle'
#path = args.path
path = '../out_files'

if len(ftype) < 1:
    print(f'No ftype given. Usage: python hpc_merge_files.py --.pickle')
    exit()

if len(path) < 1:
    print(f'No path given. Assuming work folder \'.\'. Usage: python hpc_merge_files.py --path=../out_files')
    exit()

print('Processing files...')
# Specify the directory containing the .pickle files
directory_path = path#'/path/to/your/pickle/files'

# Collect data from all .pickle files in the directory
all_data = pickle_reader.collect_data_from_directory(directory_path, ftype)

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(all_data)

# Display the DataFrame
print("DataFrame created from pickled dictionaries:")
print(df)

# Optionally, save the DataFrame to a CSV file
# Write DataFrame to a csv file
ts = datetime.now().strftime('%Y-%m-%d %H_%M')
csv_out = ts + ' merged_pickles' + '.csv'
csv_out_path = os.path.join(path, csv_out)
df.to_csv(csv_out_path, index=False)
print('DataFrame saved to ' + csv_out_path)


