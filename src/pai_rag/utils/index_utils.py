import os
import shutil
from loguru import logger


def delete_file(file_path):
    try:
        os.remove(file_path)
        logger.info(f"File {file_path} successfully deleted.")
    except FileNotFoundError:
        logger.error(f"File {file_path} does not exist.")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")


def delete_dir(folder_path):
    try:
        shutil.rmtree(folder_path)
        logger.info(f"Folder {folder_path} and its contents successfully deleted.")
    except FileNotFoundError:
        logger.error(f"Folder {folder_path} does not exist.")
    except Exception as e:
        logger.error(f"Error deleting folder {folder_path}: {e}")


def del_index_dir(index_name):
    if index_name == "default_index":
        destination_folder = os.path.join("localdata", "storage")
    else:
        destination_folder = os.path.join("localdata", index_name)
    delete_dir(destination_folder)


def write_markdown_to_parse_dir(md_content, file_name, parse_dir):
    destination_md_file = f"{parse_dir}/{file_name}.md"
    try:
        with open(destination_md_file, "w", encoding="utf-8") as md_file:
            md_file.write(md_content)
        print(f"成功写入{destination_md_file}")
    except IOError as e:
        print(f"写入文件时出错: {e}")


def copy_original_files_to_parse_dir(file_path, parse_dir):
    try:
        if os.path.isfile(file_path):
            shutil.copy(file_path, parse_dir)
            print(f"已复制: {file_path} 到 {parse_dir}")
        else:
            print(f"源文件不存在: {file_path}")
    except Exception as e:
        print(f"复制文件时出错: {file_path} -> {e}")
