import pickle
import os

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