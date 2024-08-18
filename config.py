# fraud_detection/config.py
import sys
import os


def add_src_to_sys_path():
    current_directory = os.getcwd()

    src_directory = os.path.abspath(os.path.join(current_directory, "..", "src"))
    if src_directory not in sys.path:
        sys.path.append(src_directory)

    print(f"src added to sys.path: {src_directory}")


add_src_to_sys_path()
