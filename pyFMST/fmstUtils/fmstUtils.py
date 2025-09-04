import os
import shutil
import numpy as np
import tempfile

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

def create_file_from_template(target_path: str, template_dir:str, template_file: str):

    template_file_path = os.path.join(template_dir ,template_file)
    
    deploy_file(template_file_path, target_path)

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
            grid_data.extend(values)

    total_vals = len(grid_data)
    expected = rows * cols

    if total_vals == expected:
        grid_array = np.array(grid_data).reshape(rows, cols)
    elif total_vals == expected * 2:
        grid_array = np.array(grid_data).reshape(rows, cols, 2)
    else:
        raise ValueError(f"Unexpected number of values: {total_vals}, expected {expected} or {expected*2}")

    return grid_basis, grid_step, grid_array

def process_file(file_path, block_start, block_spec, update_params=None):
    """
    Generic function to read, update, and overwrite parameter blocks in a file.

    Args:
        file_path (str): Path to the file.
        block_spec (dict): Dictionary defining parameter names and their types.
                           Example: {"grid_dicing": (int, int), "earth_radius": float}.
        block_start (int): The 0-indexed line number where the parameter block starts.
        update_params (dict): Dictionary of parameters to update, e.g., {"grid_dicing": (10, 10)}.

    Returns:
        dict: The parsed parameter block.
    """
    # Initialize the dictionary for parsed parameters
    params = {name: None for name in block_spec}
    params_deco = {name: None for name in block_spec}

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse the block
    for i, (name, param_type) in enumerate(block_spec.items()):
        line_index = block_start + i
        if isinstance(param_type, tuple):  # For tuples (e.g., (int, int))
            params[name] = tuple(map(param_type[0], lines[line_index].split()))
        else:  # For single types (e.g., int, float, str)
            params[name] = param_type(lines[line_index].strip().split()[0])
            params_deco[name] = " ".join(lines[line_index].strip().split()[1:])
            

    # Update parameters if specified
    if update_params:
        for key, value in update_params.items():
            if key in params:
                params[key] = value

        # Overwrite the lines in the file
        for i, (name, param_type) in enumerate(block_spec.items()):
            line_index = block_start + i
            if name in update_params:
                if isinstance(param_type, tuple):
                    lines[line_index] = "    ".join(map(str, params[name])) + "\n"
                else:
                    lines[line_index] = str(params[name]) + "     " + str(params_deco[name]) + "\n"
            else:
                pass


        # Write back to the file
        with open(file_path, "w") as file:
            file.writelines(lines)

def backup_files(files, temp_dir=None):
    """
    Backs up a list of files to a temporary directory.

    Args:
        files: A list of file paths to be backed up.
        temp_dir: Optional path to the temporary directory. If None, a temporary directory is created.

    Returns:
        The path to the temporary directory.
    """

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    for file in files:
        shutil.copy2(file, os.path.join(temp_dir, os.path.basename(file)))

    return temp_dir

def restore_files(temp_dir, files):
    """
    Restores a list of files from a temporary directory.

    Args:
        temp_dir: The path to the temporary directory.
        files: A list of file paths to be restored.
    """

    for file in files:
        temp_file = os.path.join(temp_dir, os.path.basename(file))
        if os.path.exists(temp_file):
            shutil.copy2(temp_file, file)

