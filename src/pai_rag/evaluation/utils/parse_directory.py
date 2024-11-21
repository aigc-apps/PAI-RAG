import os


def list_files_in_directory(directory_path):
    # 遍历目录树
    full_paths = []
    for dirpath, _, filenames in os.walk(directory_path):
        # 打印当前目录中的所有文件
        for filename in filenames:
            # 获取文件的完整路径
            full_path = os.path.join(dirpath, filename)
            full_paths.append(full_path)
    return full_paths
