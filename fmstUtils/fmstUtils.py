import os
import shutil
import numpy as np

def deploy_file(file_path: str, target_path: str):
    """
    Copies a default file from the repository to the target path.

    Args:
        file_path (str): Path to the repository's template file.
        target_path (str): Directory to copy the file into.
    """

    # Ensure the default file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Default file not found: {file_path}")

    # Ensure the target directory exists
    with open(target_path, 'w') as fp:
        pass

    # Copy the file to the target location
    shutil.copy(file_path, target_path)

def create_grid_cfg_file(target_path: str):

    template_file = os.path.join(os.path.dirname(__file__), '..', 'templates','grid2dss.in')
    deploy_file(template_file, target_path)


def read_grid_file(file_path: str):
    grid_data = []

    with open(file_path, 'r') as infile:
        lines = infile.readlines()

        dims = list(map(int, lines[0].split()))
        rows, cols = dims[0] + 2, dims[1] + 2

        lat_lon_basis = list(map(float, lines[1].split()))
        grid_basis = {'latitude': lat_lon_basis[0], 'longitude': lat_lon_basis[1]}

        grid_step = list(map(float, lines[2].split()))

        data_lines = lines[3:]
        for line in data_lines:
            values = list(map(float, line.split()))
            if values:
                grid_data.append(values)

    # Convert to the specified grid dimensions
    grid_array = np.array(grid_data).reshape(rows, cols, 2)

    return grid_basis, grid_step, grid_array
