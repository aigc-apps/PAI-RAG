import os


def list_files_in_directory(directory_path):
    full_paths = []
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            full_paths.append(full_path)
    return full_paths
