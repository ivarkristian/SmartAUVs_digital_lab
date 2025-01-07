# %%
import pandas as pd
import importlib
import os
import glob

# %%
importlib.reload()

# %%
root_dir = "../HUGIN_chemical_data_horten_2024"
missions_2024 = glob.glob(os.path.join(root_dir, '*'))
#missions_2024 = os.listdir(root_dir)
missions_2024.sort()

# %%
sensor_data_folders = ['HydroC-CO2', 'ADAMModule-EHpH', 'AADIO2']
sensor_data_file_types = ['txt', 'log', 'txt']

# %%
def concatenate_files(mission_path, sensor, filetype, outfile=None):
    """
    Concatenate all text files of type `filetype` from the directory `dir_path`
    into a single file `outfile`, sorted by filename in alphabetical order.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the files to concatenate.
    filetype : str
        The file extension/type (e.g. 'txt') of the files to concatenate.
    outfile : str
        Path to the resulting output file where contents are written.
    """
    # Create a search pattern for files matching *.filetype in dir_path
    dir_path = mission_path + '/' + sensor
    pattern = os.path.join(dir_path, f"*.{filetype}")

    # Find all matching files and sort them alphabetically
    file_list = sorted(glob.glob(pattern))

    if outfile is None:
        splitted_path = dir_path.split('/')
        outfile = mission + '/' + splitted_path[-1] + '.' + filetype

    with open(outfile, "w", encoding="utf-8") as out_f:
        for file_path in file_list:
            if file_path.endswith('dir.txt') is not True:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    out_f.write(in_f.read())
                    # Optionally add a newline or other separator between files
                    # out_f.write("\n")

# %%
# Concatenate files
# Seems like the first pH-file in every mission has a strange character in the first line, remove manually
for mission in missions_2024:
    for i, sensor in enumerate(sensor_data_folders):
        concatenate_files(mission, sensor, sensor_data_file_types[i])

# %%
def parse_EHpH_file_to_df(file_path, column_names):
    """
    Parses the given file, where each line looks like:
    1717669805050450 >+03.004+02.907+000.00+000.00+000.00+00.000+00.000+00.000
    Returns a pandas DataFrame with columns [timestamp, col1, col2, ...].
    """
    rows = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Split into timestamp and the part with '>+...' data
            parts = line.split(' ', maxsplit=1)
            if len(parts) != 2:
                continue
            
            timestamp_str = parts[0][:11]
            numeric_str = parts[1]  # e.g. ">+03.004+02.907+000.00..."
            
            # Remove the leading '>' if present
            if numeric_str.startswith('>'):
                numeric_str = numeric_str[1:]
            
            # Split by '+' to get each numeric chunk
            # Example result: ["03.004", "02.907", "000.00", "000.00", "000.00", "00.000", "00.000", "00.000"]
            chunks = numeric_str.split('+')
            
            # Filter out any empty strings if there's a leading '+' or trailing '+'
            chunks = [c for c in chunks if c.strip()]
            
            # Convert to floats
            values = [float(c) for c in chunks]

            values_sliced = values[:len(column_names) - 1]
            
            # Convert timestamp to int (or str if you prefer)
            timestamp = float(timestamp_str)/10
            
            # Build one row with [timestamp, val1, val2, ...]
            row = [timestamp] + values_sliced
            rows.append(row)
    
    # Build a DataFrame from the rows
    # Define column names: first is 'timestamp', then col1, col2, ...
    col_names = column_names
    df = pd.DataFrame(rows, columns=col_names)
    
    return df

def read_file_to_df(filename, column_names=None):
    """
    Reads a whitespace-delimited file where:
    - The first column is a timestamp.
    - The remaining columns are numeric values.
    Returns a pandas DataFrame.
    """
    # Read the file, splitting on whitespace, no headers
    df = pd.read_csv(filename, delim_whitespace=True, header=None)

    # Give meaningful names to columns:
    # Adjust the column names based on your data's meaning
    if column_names:
        df.columns = column_names
    else:
        old_name = df.columns[0]
        df.rename(columns={old_name: 'ts'}, inplace=True)
        
    
    # Convert 'timestamp' to float (or int/datetime if needed)
    df[df.columns[0]] = df[df.columns[0]].astype(float)

    # Convert other columns to float
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)

    return df

# Example usage:
# df = read_file_to_df("path_to_your_file.txt")
# print(df.head())

# Example usage:
# file_path = "path_to_your_data.txt"
# df = parse_file_to_df(file_path)
# print(df.head())

# %%
# Import to dataframes and sort by time

for mission in missions_2024:
    files = [f for f in os.listdir(mission) if (os.path.isfile(mission + '/' + f) and (f.startswith('.') is not True))]
    for file in files:
        if file.endswith('.csv') is not True:
            print(f'file: {file}')
            if 'AADIO2' in file:
                df = read_file_to_df(mission + '/' + file, column_names=['ts', 'Latitude', 'Longitude', 'Depth', 'col5', 'col6', 'col7'])
            elif 'ADAMModule-EHpH' in file:
                df = parse_EHpH_file_to_df(mission + '/' + file, column_names=['ts', 'pH', 'eH'])
            elif 'HydroC-CO2' in file:
                df = read_file_to_df(mission + '/' + file, column_names=['ts', 'Latitude', 'Longitude', 'Depth', 'col5', 'col6', 'col7'])
            elif 'navpos' in file:
                df = read_file_to_df(mission + '/' + file)
            elif 'ctd' in file:
                df = read_file_to_df(mission + '/' + file)
            elif 'vel' in file:
                df = read_file_to_df(mission + '/' + file)
            else:
                df = None
                
            if df is not None:
                # Sort by timestamp
                df.sort_values(by="ts", ascending=True, inplace=True)

                # Save to .hdf
                df.to_csv(mission + '/' + file.split('.')[0] + '.csv', mode='w')

            
# EHpH


# %%

# %%
