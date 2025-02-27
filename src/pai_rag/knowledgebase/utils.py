import os
import shutil
from loguru import logger
from pai_rag.knowledgebase.constants import DEFAULT_KNOWLEDGE_PATH


def create_new_index_dir(index_name):
    destination_folder = os.path.join(DEFAULT_KNOWLEDGE_PATH, index_name)
    destination_docs_folder = os.path.join(destination_folder, "docs")
    destination_index_folder = os.path.join(destination_folder, ".index")
    destination_index_logs_folder = os.path.join(destination_folder, ".logs")
    try:
        os.makedirs(destination_folder, exist_ok=True)
        os.makedirs(destination_docs_folder, exist_ok=True)
        os.makedirs(destination_index_folder, exist_ok=True)
        os.makedirs(destination_index_logs_folder, exist_ok=True)
        logger.info(f"目录 '{destination_folder}' 及其子目录已成功创建或已存在。")
    except Exception as e:
        logger.error(f"创建目录时发生错误: {e}")


def del_index_dir(index_name):
    destination_folder = os.path.join("localdata", index_name)
    try:
        shutil.rmtree(destination_folder)
        print(
            f"Successfully deleted the empty directory and its contents: {destination_folder}"
        )
    except OSError as e:
        print(f"Error: {destination_folder} : {e.strerror}")


def copy_input_files_to_destination(file_list, index_name):
    destination_folder = os.path.join(DEFAULT_KNOWLEDGE_PATH, index_name)

    for file_path in file_list:
        try:
            if os.path.isfile(file_path):
                shutil.copy(file_path, destination_folder)
                print(f"已复制: {file_path} 到 {destination_folder}")
            else:
                print(f"源文件不存在: {file_path}")
        except Exception as e:
            print(f"复制文件时出错: {file_path} -> {e}")
