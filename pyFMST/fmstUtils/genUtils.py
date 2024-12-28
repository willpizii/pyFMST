import os

def check_file_exists(path):
    if not os.path.exists(path):
        raise RuntimeError("Specified file path does not exist!")
    else:
        return True
