# %%
import pandas as pd
import importlib
import os
import glob
import numpy as np

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
    df = pd.read_csv(filename, sep='\s+', header=None)

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
    print(f'mission: {mission}')
    files = [f for f in os.listdir(mission) if (os.path.isfile(mission + '/' + f) and (f.startswith('.') is not True))]
    for file in files:
        if file.endswith('.csv') is not True:
            print(f'file: {file}')
            if 'AADIO2' in file:
                df = read_file_to_df(mission + '/' + file, column_names=['ts', 'Latitude', 'Longitude', 'Depth', 'Altitude', 'PartialPressure_mbar', 'O2_sat_pct'])
            elif 'ADAMModule-EHpH' in file:
                df = parse_EHpH_file_to_df(mission + '/' + file, column_names=['ts', 'pH', 'eH'])
            elif 'HydroC-CO2' in file:
                df = read_file_to_df(mission + '/' + file, column_names=['ts', 'Latitude', 'Longitude', 'Depth', 'Flags', 'Temperature', 'PartialPressure'])
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
            else:
                print('(skipped!)')

            
# You now have .csv files for every sensor, navpos, vel and ctd in the mission directories.
# There are also merged sensor data files, these may be .txt or .log (or even something else)
# depending on the original format.

# %%
def merge_csv_files(directory):
    """
    Reads all CSV files in 'directory', identifies 'navpos.csv' as the reference timeline
    (retaining only its timestamp column), and merges data from other files by nearest
    timestamp. The final DataFrame is returned.
    """
    # Find all .csv files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    # Identify the navpos file
    navpos_file = None
    for f in csv_files:
        if os.path.basename(f).lower() == 'navpos.csv':
            navpos_file = f
            break

    if navpos_file is None:
        raise FileNotFoundError("No 'navpos.csv' file found in the given directory.")

    # Read navpos.csv, keep only timestamp column
    df_navpos = pd.read_csv(navpos_file, header=0, index_col=0)
    # The second column is 'ts'; rename it to 'timestamp'
    df_navpos.rename(columns={'ts': 'timestamp'}, inplace=True)
    # Convert 'timestamp' to a proper datetime if needed
    # e.g. df_navpos['timestamp'] = pd.to_datetime(df_navpos['timestamp'], unit='s')

    # Drop any other columns, keeping only 'timestamp'
    df_navpos = df_navpos[['timestamp']]

    # Sort by 'timestamp' for merge_asof to work
    df_navpos.sort_values('timestamp', inplace=True)
    df_navpos.reset_index(drop=True, inplace=True)

    # We'll store the merged result here
    merged_df = df_navpos.copy()

    # Merge all other files
    for f in csv_files:
        print(f'file: {f}')
        #if f == navpos_file:
        #    continue  # Skip the navpos file itself

        # Use the file's base name (without extension) as prefix
        basename = os.path.splitext(os.path.basename(f))[0]

        # Read the CSV, ignoring the first column index
        df_other = pd.read_csv(f, header=0, index_col=0)

        # The second column is 'ts'
        # Convert it to 'timestamp' for merging
        df_other.rename(columns={'ts': 'timestamp'}, inplace=True)

        # Convert 'timestamp' to numeric or datetime if needed
        # e.g. df_other['timestamp'] = pd.to_datetime(df_other['timestamp'], unit='s')

        # Sort by 'timestamp' for merge_asof
        df_other.sort_values('timestamp', inplace=True)
        df_other.reset_index(drop=True, inplace=True)

        # Prefix other columns with the filename (excluding 'timestamp')
        cols_to_prefix = [c for c in df_other.columns if c != 'timestamp']
        df_other.rename(columns={c: f"{basename}_{c}" for c in cols_to_prefix}, inplace=True)

        # Perform an asof merge on the nearest timestamp
        merged_df = pd.merge_asof(
            merged_df.sort_values('timestamp'),
            df_other.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )

    # Final sorting by timestamp just to be consistent
    merged_df.sort_values('timestamp', inplace=True)

    return merged_df

# %%
# Merge .csv files into one data frame
merged_dfs = []
for mission in missions_2024:
    print(f'mission: {mission}')
    merged_df = merge_csv_files(mission)
    # Convert from seconds since epoch to datetime
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], unit='s').dt.round('1s')
    merged_dfs.append(merged_df)

# Concatenate merged_dfs
combined_df = pd.concat(merged_dfs, axis=0, ignore_index=True)

# %%
# Rename columns (should add units here)
name_mappings = {'ctd_1': 'CTD_conductivity',
                 'ctd_2': 'CTD_temperature',
                 'ctd_3': 'CTD_pressure',
                 'ctd_4': 'CTD_salinity',
                 'vel_1': 'FA_BOTM_VEL',
                 'vel_2': 'PS_BOTM_VEL',
                 'vel_3': 'VERT_BOTM_VEL',
                 'vel_5': 'FA_WTR_VEL',
                 'vel_6': 'PS_WTR_VEL',
                 'vel_7': 'VERT_WTR_VEL',
                 'navpos_11': 'Latitude',
                 'navpos_12': 'Longitude',
                 'navpos_15': 'Depth',
                 }

combined_df.rename(columns=name_mappings, inplace=True)

# %%
# Drop columns that are not interesting
retain_columns = ['timestamp', 'CTD_conductivity', 'CTD_temperature', 'CTD_pressure', 'CTD_salinity', 'FA_BOTM_VEL',
                  'PS_BOTM_VEL', 'VERT_BOTM_VEL', 'FA_WTR_VEL', 'PS_WTR_VEL', 'VERT_WTR_VEL', 
                  'Latitude', 'Longitude', 'Depth', 'AADIO2_PartialPressure_mbar', 'AADIO2_O2_sat_pct', 
                  'ADAMModule-EHpH_pH', 'ADAMModule-EHpH_eH', 'HydroC-CO2_PartialPressure']
df = combined_df[retain_columns]

# %%
# Compute xyz in meters based on latitude, longitude and depth columns
def add_local_xy(df, lat_col='Latitude', lon_col='Longitude'):
    """
    Adds two new columns to df: 'Lat_m' and 'Lon_m',
    which represent distances in meters from the southernmost, westernmost coordinate.
    Assumes lat/lon in degrees, and uses a simple small-scale approximation.
    """
    # 1) Find reference latitude and longitude (the minimal lat/lon)
    lat0 = df[lat_col].min()
    lon0 = df[lon_col].min()

    # 2) Define approximate conversion factors
    # degrees -> meters for latitude (constant ~111.111 km/deg)
    meters_per_deg_lat = 111_111
    
    # For longitude, we use cos of a reference latitude
    # (you can also use the average latitude if you prefer)
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(lat0))
    
    # 3) Create new columns for local x, y (or easting/northing)
    df['Lat_m'] = (df[lat_col] - lat0) * meters_per_deg_lat
    df['Lon_m'] = (df[lon_col] - lon0) * meters_per_deg_lon
    
    return df

df = add_local_xy(df)
df['Depth_m'] = df['Depth']*-1

# %%
# Save to csv
df.to_csv(root_dir + '/merged_missions.csv')

