import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from .fmstUtils import fmstUtils

class fmst:

    def __init__(self,
                 path: str):
        """
        Initiliazes the fmst class

        Requires a file path for operations and files to be built within
        """

        if not path:
            raise ValueError("The path must be provided.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")

        self.path = path
        self.initial_velocity = None
        self.region = None

        # initiliase grid config file

        self.__mkmodel_dir = os.path.join(self.path, 'mkmodel')
        self.__grid_path = os.path.join(self.path, 'mkmodel', 'grid2dss.in')

        if not os.path.exists(self.__grid_path):
            self.__mkmodel_dir = os.path.join(self.path, 'mkmodel')
            if not os.path.exists(self.__mkmodel_dir):
                os.makedirs(self.__mkmodel_dir)

            fmstUtils.create_grid_cfg_file(self.__grid_path)

    def set_background_vel(self, v):
        """
        Sets background velocity for the input grid

        Parameters:
            v : A float, list or nparray of velocities for background velocity to be set upon
                If passed as a list or array, the mean value is taken
        """
        if isinstance(v, float):
            self.initial_velocity = v
        elif isinstance(v, list) or isinstance(v, np.array):
            self.initial_velocity = np.mean(v)
        else:
            raise TypeError(f"v expected as float, list or np.array, but found {type(v)}")


    def config_grid(self,
                    region: list,
                    latgrid: int,
                    longrid: int,
                    noise: bool=False,
                    noise_std: float,
                    noise_seed: int,
                    unc: bool=True,
                    unc_mag: float=0.3):

        """
        Configures grid for FMST to create using the builtin script grid2dss.

        Parameters:
            region (list):       A list of floats or integers specifying the region boundaries.
                        The order should be: [max_latitude, min_latitude, min_longitude, max_longitude].
            latgrid (int):      The number of grid points in the north-south (N-S) direction.
            longrid (int):      The number of grid points in the east-west (E-W) direction.
            noise (bool):       Enables or disables grid noise. Defaults to False
            noise_std (float):  Standard deviation of noise. If not set, defaults to 0.8
            noise_seed (int):   Seed for the random noise. If not set, defaults to 12324
            unc (bool):         Enables or disables a priori model covariance. Defaults to True
            unc_mag (float):    Elements of the covariance matrix. Defaults to 0.3
        """

        if not self.initial_velocity:
            raise RuntimeError("Background velocity must be set before creating a grid.")

        if not all(isinstance(coord, (int, float)) for coord in region) or len(region) != 4:
            raise ValueError("Region should be a list of four numeric values (e.g., [lat, lon]).")

        with open(self.__grid_path, 'r') as infile:
            lines = infile.readlines()

        lines[14] = f'{round(self.initial_velocity, 3):<24} c: Background velocity\n'

        lines[7] = f'{latgrid}                   c: Number of grid points in theta (N-S)\n'
        lines[8] = f'{longrid}                   c: Number of grid points in phi (E-W)\n'

        lines[9] = f'{region[0]}  {region[1]}          c: E-W range of grid (degrees)\n'
        lines[10] = f'{region[2]}  {region[3]}          c: E-W range of grid (degrees)\n'


        if noise and not noise_std:
            raise Warning("No amplitude passed for noise - defaulting to 0.8!")
            noise_std = 0.8

        if noise and not noise_seed:
            raise Warning("No seed passed for noise - defaulting to 12324!")
            noise_seed = 12324

        if noise:
            lines[18] = '1                     c: Add random structure (0=no,1=yes)\n'
            lines[19] = f'{noise_std}                  c: Standard deviation of random noise\n'
            lines[20] = f'{noise_seed}                   c: Random seed for noise generation\n'
        else:
            lines[18] = '0                     c: Add random structure (0=no,1=yes)\n'


        if unc:
            lines[24] = '1                     c: Add a priori model covariance (0=no,1=yes)?\n'
            lines[25] = f'{unc_mag}                   c: Diagonal elements of covariance matrix\n'

        else:
            lines[24] = '0                     c: Add a priori model covariance (0=no,1=yes)?\n'

        with open(self.__grid_path, 'w') as outfile:
            outfile.writelines(lines)

        self.region = region

    def create_grid(self,
                    copy_to_gridi: bool=True):

        subprocess.run('grid2dss', cwd=self.__mkmodel_dir, shell=True)

        self.__grid_file_init = os.path.join(self.path, 'mkmodel', 'grid2d.vtx')

        _,self.grid_step, self.gridi = fmstUtils.read_grid_file(self.__grid_file_init)

        if copy_to_gridi:

            self.__gridi_file = os.path.join(self.path, 'gridi.vtx')

            shutil.copy(self.__grid_file_init, self.__gridi_file)
