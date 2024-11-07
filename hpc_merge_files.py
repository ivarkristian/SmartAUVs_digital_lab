# %%
import pickle
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

def load_pickled_dicts_from_file(file_path):
    """
    Loads all pickled dictionaries from a file and returns them as a list.

    Parameters:
    - file_path: str, path to the pickle file.

    Returns:
    - data_list: list of dictionaries loaded from the file.
    """
    data_list = []
    with open(file_path, 'rb') as file_in:
        while True:
            try:
                data_dict = pickle.load(file_in)
                data_list.append(data_dict)
            except EOFError:
                break
    return data_list

def collect_data_from_directory(directory_path, ftype):
    """
    Collects dictionaries from all .pickle files in a directory.

    Parameters:
    - directory_path: str, path to the directory containing .pickle files.

    Returns:
    - all_data: list of dictionaries collected from all files.
    """
    all_data = []
    # List all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(ftype):
            print(filename)
            file_path = os.path.join(directory_path, filename)
            data_list = load_pickled_dicts_from_file(file_path)
            all_data.extend(data_list)
    return all_data


print('Processing files...')
# Specify the directory containing the .pickle files
directory_path = path#'/path/to/your/pickle/files'

# Collect data from all .pickle files in the directory
all_data = collect_data_from_directory(directory_path, ftype)

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


# %%
