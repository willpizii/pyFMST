import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from obspy import read_inventory
import pygmt
import shutil
import json
from tqdm import tqdm
import re

import sys
sys.path.append('./pyFMST')  # Add the parent directory of pyFMST to the system path

from fmstUtils import fmstUtils, genUtils

class fmst:

    def __init__(self,
                 path: str,
                 templates: str):
        """
        Initiliazes the fmst class

        Requires a file path for operations and files to be built within
        """

        if not path:
            raise ValueError("The path must be provided.")

        genUtils.check_file_exists(path)

        self.path = path

        genUtils.check_file_exists(templates)

        self.templates_dir = templates

        self.initial_velocity = None
        self.region = None

        self.refined = False

        # initiliase grid config file

        self.__mkmodel_dir = os.path.join(self.path, 'mkmodel')
        self.__grid_path = os.path.join(self.path, 'mkmodel', 'grid2dss.in')

        if not os.path.exists(self.__grid_path):
            self.__mkmodel_dir = os.path.join(self.path, 'mkmodel')
            if not os.path.exists(self.__mkmodel_dir):
                os.makedirs(self.__mkmodel_dir)

            fmstUtils.create_file_from_template(self.__grid_path, self.templates_dir, 'grid2dss.in')

    def set_background_vel(self, v):
        """
        Sets background velocity for the input grid

        Parameters:
            v : A float, list or nparray of velocities for background velocity to be set upon
                If passed as a list or array, the mean value is taken
        """
        if isinstance(v, float):
            self.initial_velocity = v
        elif isinstance(v, list) or isinstance(v, np.ndarray):
            self.initial_velocity = np.mean(v)
        else:
            raise TypeError(f"v expected as float, list or np.array, but found {type(v)}")

    def set_region(self, region:list):

        """
        Parameters:
            region (list):       A list of floats or integers specifying the region boundaries.
                                 The order should be: [max_latitude, min_latitude, min_longitude, max_longitude].
        """

        # Check region list validity

        if not region[0] > region[1] or not region[2]<region[3] or len(region) != 4:
            raise ValueError("Specified region is incorrectly formatted.")

        self.region = region

    def config_grid(self,
                    latgrid: int,
                    longrid: int,
                    noise: bool=False,
                    noise_std: float=0.8,
                    noise_seed: int=12324,
                    unc: bool=True,
                    unc_mag: float=0.3):

        """
        Configures grid for FMST to create using the builtin script grid2dss.

        Parameters:
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

        with open(self.__grid_path, 'r') as infile:
            lines = infile.readlines()

        lines[14] = f'{round(self.initial_velocity, 3):<24} c: Background velocity\n'

        lines[7] = f'{latgrid}                   c: Number of grid points in theta (N-S)\n'
        lines[8] = f'{longrid}                   c: Number of grid points in phi (E-W)\n'

        lines[9] = f'{self.region[0]}  {self.region[1]}          c: N-S range of grid (degrees)\n'
        lines[10] = f'{self.region[2]}  {self.region[3]}          c: E-W range of grid (degrees)\n'


        if noise and not noise_std:
            raise Warning("No amplitude passed for noise - defaulting to 0.8!")


        if noise and not noise_seed:
            raise Warning("No seed passed for noise - defaulting to 12324!")


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

    def create_grid(self,
                    copy_to_gridi: bool=True):

        subprocess.run('grid2dss', cwd=self.__mkmodel_dir, shell=True)

        self.__grid_file_init = os.path.join(self.path, 'mkmodel', 'grid2d.vtx')

        _,self.grid_step, self.gridi = fmstUtils.read_grid_file(self.__grid_file_init)

        if copy_to_gridi:

            self.__gridi_file = os.path.join(self.path, 'gridi.vtx')

            shutil.copy(self.__grid_file_init, self.__gridi_file)

    def load_stations(self, station_path: str):
        genUtils.check_file_exists(station_path)

        __ext = os.path.splitext(station_path)[-1]
        __sta_cols = ['network', 'station', 'lat', 'lon', 'elev']

        # Initialize as an empty list to collect rows
        rows = []

        if __ext == '.xml':
            __inv = read_inventory(station_path)

            for network in __inv:
                for station in network:
                    # Add rows to the list directly
                    rows.append([network.code, station.code, station.latitude, station.longitude, station.elevation])

        elif __ext == '.csv':
            __stadf = pd.read_csv(station_path, usecols=__sta_cols)
            # Append rows to list from the CSV
            rows.extend(__stadf.values.tolist())

        else:
            raise ValueError("Supported formats are inventoryxml .xml and .csv!")

        # Convert the list of rows into a DataFrame
        __stadf = pd.DataFrame(rows, columns=__sta_cols)

        # Filter based on region
        __stadf = __stadf[(__stadf['lon'] >= self.region[2]) &
                          (__stadf['lon'] <= self.region[3]) &
                          (__stadf['lat'] >= self.region[1]) &
                          (__stadf['lat'] <= self.region[0])]

        # Assign to self.stations after filtering
        self.stations = __stadf
        self.station_count = int(len(self.stations))


    def load_velocity_pairs(self,
                            velocity_pairs_path: str,
                            phase_vel: float):

        genUtils.check_file_exists(velocity_pairs_path)

        __ext = os.path.splitext(velocity_pairs_path)[-1]

        if __ext != '.json':
            raise ValueError("Only json paired velocity inputs are supported!")

        with open(velocity_pairs_path, 'r') as file:
            data = json.load(file)

        self.velocity_pairs = {}

        for item, value in data.items():
            if phase_vel in value[0]:
                index_pf = value[0].index(phase_vel)
                self.velocity_pairs[item] = value[1][index_pf]

    def read_station_pairs(self,
                           station_pairs_path: str,
                           drop: bool=False):

        __sta_pairs = pd.read_csv(station_pairs_path)

        __sta_pairs['vel'] = None

        for item, value in self.velocity_pairs.items():
            sta1 = item.split("_")[1]
            sta2 = item.split("_")[-1]

            __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'vel'] = value
            __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'loc1'] = int(self.stations.index[self.stations['station'] == sta1].tolist()[0])
            __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'loc2'] = int(self.stations.index[self.stations['station'] == sta2].tolist()[0])

        if drop:
            __used_stations = list(pd.unique(__sta_pairs[['loc1', 'loc2']].values.ravel()))

            __orig_stations = len(self.stations)

            self.stations = self.stations[self.stations.index.isin(__used_stations)]
            self.stations.reset_index(drop=True, inplace=True)

            self.station_count = int(len(self.stations))

            print("Removed", __orig_stations - len(self.stations), "unused stations!")

            for item, value in self.velocity_pairs.items():
                sta1 = item.split("_")[1]
                sta2 = item.split("_")[-1]

                __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'vel'] = value
                __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'loc1'] = int(self.stations.index[self.stations['station'] == sta1].tolist()[0])
                __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'loc2'] = int(self.stations.index[self.stations['station'] == sta2].tolist()[0])

        __sta_pairs['gcm'] /= 1000 # converts from metres to kilometres

        __sta_pairs['tt'] = __sta_pairs['gcm'] / __sta_pairs['vel']

        __sta_pairs_clean = __sta_pairs.dropna(subset=['tt']).reset_index(drop=True)

        self.station_pairs = __sta_pairs_clean[['loc1','loc2','tt']]

        for idx, station in enumerate(zip(__sta_pairs_clean['station1'].tolist(), __sta_pairs_clean['station2'].tolist())):
            __sta_pairs_clean.at[idx, 'lat1'] = self.stations[self.stations['station'] == station[0]]['lat'].values[0]
            __sta_pairs_clean.at[idx, 'lat2'] = self.stations[self.stations['station'] == station[1]]['lat'].values[0]
            __sta_pairs_clean.at[idx, 'lon1'] = self.stations[self.stations['station'] == station[0]]['lon'].values[0]
            __sta_pairs_clean.at[idx, 'lon2'] = self.stations[self.stations['station'] == station[1]]['lon'].values[0]

        self.station_pairs_complete = __sta_pairs_clean

    def refine_station_pairs(self,
                           method: str=None,
                           arg: float=None):


        if not self.refined:
            self.station_pairs_original = self.station_pairs_complete

        __sta_pairs = self.station_pairs_original.copy()

        __original_len = len(__sta_pairs.dropna(subset='vel'))

        if method == 'std':
            __mean_vel = np.mean(__sta_pairs['vel'])
            __std_vel = np.std(__sta_pairs['vel'])

            __sta_pairs = __sta_pairs[(__sta_pairs['vel'] >= __mean_vel - arg * __std_vel) & (__sta_pairs['vel'] <= __mean_vel + arg * __std_vel)]

        elif method == 'abs':
            __lower_bound = np.percentile(__sta_pairs['vel'], arg)
            __upper_bound = np.percentile(__sta_pairs['vel'], 100 - arg)

            __sta_pairs = __sta_pairs[(__sta_pairs['vel'] >= __lower_bound) & (__sta_pairs['vel'] <= __upper_bound)]

        else:
            raise ValueError("Supported methods of refine are 'abs' (absolute) or 'std' (standard deviation)")

        print("Discarded", __original_len - len(__sta_pairs), "velocity pairs with specified refine method (",__original_len , len(__sta_pairs),")")

        __sta_pairs_clean = __sta_pairs.dropna(subset=['tt']).reset_index(drop=True)

        self.station_pairs_complete = __sta_pairs_clean

        self.refined = True

    def create_sources(self):

        __sources = self.stations[['lat','lon']]

        __sources.to_csv(os.path.join(self.path,'sources.dat'), sep=r" ", header=None, index=False)

        with open(os.path.join(self.path,'sources.dat'), 'r') as file:
            lines = file.readlines()

        lines.insert(0, f"   {self.station_count}\n")

        with open(os.path.join(self.path,'sources.dat'), 'w') as file:
            file.writelines(lines)

        __sources.to_csv(os.path.join(self.path,'receivers.dat'), sep=r" ", header=None, index=False)

        with open(os.path.join(self.path,'receivers.dat'), 'r') as file:
            lines = file.readlines()

        lines.insert(0, f"   {self.station_count}\n")

        with open(os.path.join(self.path,'receivers.dat'), 'w') as file:
            file.writelines(lines)

    def create_otimes(self,
                      unc: float=0.1,
                      write: bool=True):

        __num_paths = self.station_count ** 2

        paths = [[0,0.,unc]] * __num_paths

        for _, row in self.station_pairs.iterrows():
            idx = int(row['loc1'] * self.station_count + row['loc2'])
            paths[idx] = [1,row['tt'],0.1]

        self.otimes = paths

        with open(os.path.join(self.path, "otimes.dat"), "w") as file:
            for _ in paths:
                # Convert each list to a space-separated string
                file.write(" ".join(map(str, _)) + "\n")

    def config_ttomoss(self,
                       init: bool=False,
                       fm2dss: dict=None,
                       misfitss: dict=None,
                       subinvss: dict=None,
                       subiter: int=None,
                       ttomoss: int=None):

        """
        Configure TTOMOSS inversion files.

        If no configuration dict, or no argument within a dict, is passed, the value is unchanged.

        Args:
            init (bool): If True, creates configuration files from default templates.
                         If False, will rewrite onto existing files
            fm2dss (dict): Parameters for the `fm2dss.in` file, must be passed in following format:

                            "grid_dicing": (int, int),
                            "source_grid_refinement": int,
                            "refinement_level_extent": (int, int),
                            "earth_radius": float,
                            "scheme": int,
                            "narrow_band_size": float

            misfitss (dict): Parameters for the `misfitss.in` file, must be passed in following format:

                             "dicing": (int, int),
                             "earth_radius":float

            subinvss (dict): Parameters for the `subinvss.in` file, must be passed in following format:

                            "damping":float,                    [epsilon]
                            "subspace_dimension":int,
                            "2nd_derivative_smoothing":int,     [0=no, 1=yes]
                            "smoothing":float,                    [eta]
                            "latitude_account":int,             [0=no, 1=yes]
                            "frac_G_size": float

            subiter (int): Value to write to `subiter.in` file.
            ttomoss (int): Value to write to `ttomoss.in` file.

        """

        if init:
            __config_files = ['fm2dss.in', 'ttomoss.in', 'misfitss.in', 'subinvss.in', 'subiter.in', 'residualss.in']

            for _ in __config_files:
                fmstUtils.create_file_from_template(os.path.join(self.path, _),self.templates_dir, _)

        if subiter:
            with open(os.path.join(self.path, 'subiter.in'), 'w') as file:
                file.write(subiter)

        if ttomoss:
            with open(os.path.join(self.path, 'ttomoss.in'), 'w') as file:
                file.write(ttomoss)

        if subinvss:
            __params = {'damping':float,
                        'subspace_dimension':int,
                        '2nd_derivative_smoothing':int,
                        'smoothing':float,
                        'latitude_account':int,
                        'frac_G_size': float}

            fmstUtils.process_file(os.path.join(self.path, 'subinvss.in'), 11, __params, subinvss)

        if misfitss:
            __params = {'dicing': (int, int),
                        'earth_radius':float}

            fmstUtils.process_file(os.path.join(self.path, 'misfitss.in'), 6, __params, misfitss)

        if fm2dss:
            __params = {
                "grid_dicing": (int, int),
                "source_grid_refinement": int,
                "refinement_level_extent": (int, int),
                "earth_radius": float,
                "scheme": int,
                "narrow_band_size": float
            }

            fmstUtils.process_file(os.path.join(self.path, 'fm2dss.in'), 7, __params, fm2dss)

    def run_ttomoss(self,
                   verbose: bool=False,
                   overwrite: bool=True):

        _ = subprocess.run('ttomoss', cwd=self.path, shell=True, check=True, capture_output=True, text=True)

        if verbose:
            print(_.stdout)

        with open(os.path.join(self.path, 'residuals.dat'), 'r') as file:
            lines = [[float(i) for i in line.split()] for line in file]

        if overwrite:
            self.residual_o = lines[0][0]
            self.variance_o = lines[0][1]
            self.residual_f = lines[-1][0]
            self.variance_f = lines[-1][1]

        else:
            return([lines[0][0], lines[0][1], lines[-1][0], lines[-1][1]])

    def run_tslicess(self, verbose: bool=False):

        if not os.path.exists(os.path.join(self.path, 'gmtplot')):
            os.makedirs(os.path.join(self.path, 'gmtplot'))

        if not os.path.exists(os.path.join(self.path, 'gmtplot', 'tslicess.in')):
            fmstUtils.create_file_from_template(os.path.join(self.path, 'gmtplot', 'tslicess.in'),self.templates_dir, 'tslicess.in')

        _ = subprocess.run('tslicess', cwd=os.path.join(self.path, 'gmtplot'), shell=True, check=True, capture_output=True, text=True)

        if verbose:
            print(_.stdout, _.stderr)

    def load_result_grid(self):

        with open(os.path.join(self.path, 'gmtplot', 'bound.gmt'), "r") as file:
            bounds = [float(line.strip()) for line in file]

        if bounds[4] < 1:
            x_coords = np.arange(bounds[0], bounds[1], bounds[4])
            y_coords = np.arange(bounds[2], bounds[3] + bounds[5], bounds[5])

        else:
            x_coords = np.arange(self.region[2], self.region[3], abs(self.region[2]-self.region[3])/bounds[4])
            y_coords = np.arange(self.region[1], self.region[0], abs(self.region[1]-self.region[0])/bounds[5])

        self.z_values = pd.read_csv(os.path.join(self.path,'gmtplot','grid2dv.z'), header=None).values.flatten()

        if len(self.z_values) != len(x_coords) * len(y_coords):
            raise ValueError(f"The number of Z values doesn't match the grid dimensions. {len(z_values)}, {len(x_coords) * len(y_coords)}")

        # Create a 2D grid for X, Y, and Z
        z_grid = self.z_values.reshape((len(x_coords), len(y_coords))).T

        # Create a mesh grid for X and Y coordinates
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Stack X, Y, and Z values into a single array for xyz2grd
        xyz_data = np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])

        self.xyz_data = xyz_data
        self.bounds = bounds

    def plot_map(self,
                 nlevels: int=11,
                 cmap: str="SCM/vik",
                 reverse_cmap: bool=True,
                 projection: str='M15c',
                 plot_tomo: bool=True,
                 plot_rays: bool=False,
                 plot_rays_v: bool=False,
                 plot_stations: bool=False,
                 plot_caption: bool=False,
                 save_fig: bool=False
                 ):

        gmt_region = [self.region[2], self.region[3], self.region[1], self.region[0]]

        orig_grid = pygmt.xyz2grd(data=self.xyz_data,
                                region=gmt_region,
                                spacing=[self.bounds[6], self.bounds[7]],
                                registration="pixel")

        fig = pygmt.Figure()

        if plot_tomo:
            median_z = np.median(self.z_values)
            max_z = np.max(self.z_values)
            min_z = np.min(self.z_values)
            z_disp = np.max([max_z-median_z, median_z-min_z])

            min_val = median_z - z_disp
            max_val = median_z + z_disp
            increment = (max_val - min_val) / nlevels

            # Generate the CPT with series and 10 levels
            cpt = pygmt.grd2cpt(
                grid=orig_grid,
                cmap=cmap,
                reverse=reverse_cmap,
                series=[min_val, max_val, increment]  # [min, max, increment]
            )

            fig.grdimage(
                grid=orig_grid,  # Input grid
                region=gmt_region,
                projection=projection,  # Mercator projection (6 inches wide)
                cmap= cpt,
                interpolation="c",
                dpi=150
            )

            fig.colorbar(cmap=cpt,
                frame=["x+lVelocity / km/s", "af"])

        if plot_stations:
            receivers = pd.read_csv(os.path.join(self.path,'gmtplot','receivers.dat'), sep="\s+", header=None)
            receivers = receivers[[1, 0]] # lat and lon are wrong way around!


            fig.plot(
                data=receivers,
                region=gmt_region,                 # Map boundaries (equivalent to $bounds)
                projection=projection,               # Map projection (equivalent to $proj)
                style="t0.3c",
                fill="white",
                pen="black"
            )

        # fig.basemap(region=region, projection="M6i", frame=["a10f5", "a10f5"])  # Frame with intervals
        fig.coast(
            shorelines=True,  # Draw coastlines
            borders=[1, 2],  # Show internal administrative boundaries and countries
            resolution="f",
            frame=True
        )

        if plot_rays:
            fig.plot(
                data=os.path.join(self.path,'gmtplot','rays.dat'),          # Input data file
                region=gmt_region,            # Map boundaries (equivalent to $bounds)
                projection=projection,          # Map projection (equivalent to $proj)
                pen="0.5p",               # Line width of 0.5 points (equivalent to -W0.5)
            )

        if plot_rays_v:
            # Ensure the 'vel' values are valid
            vel_min = self.station_pairs_complete["vel"].min()
            vel_max = self.station_pairs_complete["vel"].max()

            # Create a colormap based on the velocity range
            pygmt.makecpt(cmap="SCM/batlow", series=[vel_min, vel_max])

            # Define a function to compute the alpha transparency
            def compute_alpha(vel, vel_min, vel_max):
                # Calculate the distance from the middle of the range
                mid_point = (vel_min + vel_max) / 2
                alpha_value = 100 * (1 - abs(vel - mid_point) / mid_point)  # Alpha is higher for outliers
                return np.clip(alpha_value, 20, 100)  # Clip to ensure alpha stays within bounds

            # Plot rays with colors based on 'vel' and 'alpha'
            for _, row in self.station_pairs_complete.iterrows():
                alpha = compute_alpha(row['vel'], vel_min, vel_max)

                fig.plot(
                    x=[row["lon1"], row["lon2"]],
                    y=[row["lat1"], row["lat2"]],
                    pen=f"2p",
                    zvalue=row['vel'],
                    cmap=True,
                    transparency=alpha  # Apply calculated alpha
                )

        if plot_caption:
            fig.text(
                text=plot_caption,
                position="LB",
                pen="0.25p,black,solid",
                fill="white",
                font="15p",
                offset="0.5/0.5"
            )

        if save_fig:
            fig.savefig(save_path, crop=True)

        fig.show()

    def plot_hist(self,
                  use_model: bool=False,
                  mode: str='abs',
                  xlim: float=3):

        if use_model:

            with open(os.path.join(self.path, "rtravel.out"), "r") as file:
                ftimes = [list(map(float, line.strip().split())) for line in file]

        __num_paths = self.station_count ** 2

        paths = [[0,0.]] * __num_paths

        if self.refined:
            for _, row in self.station_pairs_original.iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['gcm'] / self.initial_velocity]
            alpha = 0.3
        else:
            for _, row in self.station_pairs_complete.iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['gcm'] / self.initial_velocity]

            alpha = 0.8

        itimes = paths

        try:
            assert self.otimes != None
        except:
            self.create_otimes(write=False)

        assert len(self.otimes) == len(itimes), "Mismatch in number of rows between otimes and itimes"
        assert all(len(o) - 1 == len(i) for o, i in zip(self.otimes, itimes)), "Mismatch in row dimensions"

        # Remove the last value from each sublist in otimes
        trimmed_otimes = [o[:-1] for o in self.otimes]

        if mode=="abs":
            histbins = np.linspace(-xlim,xlim,60)
            xlims = [-xlim,xlim]
            label = "Residual Time / s"

            # Perform element-wise subtraction
            rtimes = [
                [o - i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, itime_row)]
                for otime_row, itime_row in zip(trimmed_otimes, itimes)
            ]
        else:
            histbins = np.linspace(1-xlim,1+xlim,60)
            xlims=[1-xlim,1+xlim]
            label = "Residual %"

            rtimes = [
                [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, itime_row)]
                for otime_row, itime_row in zip(trimmed_otimes, itimes)
            ]

        # Extract the last value of each sublist, ignoring zeros
        last_values = [row[-1] for row in rtimes if row[-1] != 0]

        # Plot the frequency distribution
        plt.figure(figsize=(8, 6))
        plt.hist(last_values, bins=histbins, color='C0', edgecolor='black', alpha=alpha, label="original")
        plt.title("Travel Time Residuals")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xlim(xlims)

        bottom_5 = np.percentile(last_values, 5)
        top_5 = np.percentile(last_values, 95)
        plt.axvline(bottom_5, color='red', linestyle='dashed', linewidth=2, label=f'Bottom 5% ({bottom_5:.2f})')
        plt.axvline(top_5, color='green', linestyle='dashed', linewidth=2, label=f'Top 5% ({top_5:.2f})')

        mean = np.mean(last_values)
        stdev = np.std(last_values)
        plt.axvline(mean, color='grey', linestyle='dashed', linewidth=2, label=f'Mean')
        plt.axvline(mean+2*stdev, color='black', linestyle='dashed', linewidth=2, label=f'2 StDev')
        plt.axvline(mean-2*stdev, color='black', linestyle='dashed', linewidth=2)

        if self.refined:
            paths = [[0,0.]] * __num_paths

            for _, row in self.station_pairs_complete.iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['gcm'] / self.initial_velocity]
            alpha = 0.8

            itimes = paths

            try:
                assert self.otimes != None
            except:
                self.create_otimes(write=False)

            assert len(self.otimes) == len(itimes), "Mismatch in number of rows between otimes and itimes"
            assert all(len(o) - 1 == len(i) for o, i in zip(self.otimes, itimes)), "Mismatch in row dimensions"

            # Remove the last value from each sublist in otimes
            trimmed_otimes = [o[:-1] for o in self.otimes]

            if mode=="abs":
                histbins = np.linspace(-xlim,xlim,60)
                xlims = [-xlim,xlim]
                label = "Residual Time / s"

                # Perform element-wise subtraction
                rtimes = [
                    [o - i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, itime_row)]
                    for otime_row, itime_row in zip(trimmed_otimes, itimes)
                ]
            else:
                histbins = np.linspace(1-xlim,1+xlim,60)
                xlims=[1-xlim,1+xlim]
                label = "Residual %"

                rtimes = [
                    [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, itime_row)]
                    for otime_row, itime_row in zip(trimmed_otimes, itimes)
                ]

            # Extract the last value of each sublist, ignoring zeros
            last_values = [row[-1] for row in rtimes if row[-1] != 0]

            plt.hist(last_values, bins=histbins, color='C0', edgecolor='black', alpha=alpha, label="refined")

        if use_model:
            if mode=="abs":
                reltimes = [
                    [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, ftime_row)]
                    for otime_row, ftime_row in zip(trimmed_otimes, ftimes)
                ]
            else:
                reltimes = [
                    [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, ftime_row)]
                    for otime_row, ftime_row in zip(trimmed_otimes, ftimes)
                ]


            assert(len(reltimes) == len(rtimes))
            # Extract the last value of each sublist, ignoring zeros
            last_values = [row[-1] for row in reltimes if row[-1] != 0]

            # Plot the frequency distribution
            plt.hist(last_values, bins=histbins, color='pink', edgecolor='black', alpha=0.7, label="final model")

        plt.legend()

        plt.show()

    def lcurve(self,
               factor: str='smoothing',
               points: int=10,
               sample_range: list=[0.01, 100]):

        if factor not in ['smoothing', 'damping']:
            raise ValueError("factor must be one of 'smoothing' or 'damping'")

        files_to_backup = ["frechet.out", "gridc.vtx", "itimes.dat", "raypath.out", "residuals.dat", "rtravel.out", "subinvss.in", "subiter.in"]
        temp_dir = fmstUtils.backup_files(files_to_backup)

        # Perform your operation here...

        sample_space = [round(i, -int(np.floor(np.log10(abs(i))))) for i in np.logspace(np.log10(sample_range[0]), np.log10(sample_range[1]), points)]

        lcurve_dict = {}

        for s in tqdm(sample_space):
            self.config_ttomoss(subinvss={factor:s})
            result_t = self.run_ttomoss(overwrite = False)


            result_m = subprocess.run('misfitss', cwd=self.path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Check if the command executed successfully
            if result_m.returncode == 0:
                # Extracting the variance and roughness values using regular expressions
                variance_pattern = r"2 is\s*([0-9.E+-]+)"
                roughness_pattern = r"-1\) is\s*([0-9.E+-]+)$"

                # Extracting values from the output
                variance = float(re.search(variance_pattern, result_m.stdout).group(1))
                roughness = float(re.search(roughness_pattern, result_m.stdout).group(1))
            else:
                print(f"Error executing misfitss: {result.stderr}")

            if factor == 'smoothing':
                lcurve_dict[s] = [float(result_t[3]), variance]
                label_x = r"Model Variance / (km/s)$^2$"

            else:
                lcurve_dict[s] = [float(result_t[3]), roughness]
                label_x = r"Model Roughness / (kms)$^{-1}$"

        fmstUtils.restore_files(temp_dir, files_to_backup)
        shutil.rmtree(temp_dir)  # Clean up the temporary directory

        fig, ax = plt.subplots()

        ax.scatter([value[1] for value in lcurve_dict.values()],[value[0] for value in lcurve_dict.values()])
        ax.plot([value[1] for value in lcurve_dict.values()],[value[0] for value in lcurve_dict.values()], color='grey', zorder=-1, alpha=0.5)
        for key, value in lcurve_dict.items():
            ax.annotate(
                round(key, 3),
                (value[1] * 1.01, value[0] * 1.01),  # Manually offset the coordinates
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                ha='left', va='bottom'  # Justify bottom-left corner
            )

        ax.set_ylabel(r"Data Variance / s$^2$")
        ax.set_xlabel(label_x)

        plt.tight_layout()
        plt.show()
