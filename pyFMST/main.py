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
import warnings

from .fmstUtils import fmstUtils, genUtils

class fmst:
    def __init__(self,
                 path: str,
                 templates: str,
                 env=None):
        
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
        self.otimes = None

        self.refined = False

        self.env = env

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
                    latgrid: int=None,
                    longrid: int=None,
                    noise: bool=False,
                    noise_std: float=0.8,
                    noise_seed: int=12324,
                    unc: bool=True,
                    unc_mag: float=0.3,
                    checkerboard: bool=False,
                    checker_val: float=0.8,
                    checker_size: int=2,
                    checker_spacing: bool=True,
                    spike: bool=False):

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
            checkerboard (bool): Whether to apply a checkerboard to grid. Defaults to False
            checker_val (float): Checkerboard perturbation. Defaults to 0.8
            checker_size(int):  Size of checker cells, in grid cells. Defaults to 2
            checker_spacing(bool): Whether to apply spacing between checkerboard cells. Defaults to True
            spike (bool):       Whether to apply a spike test to the grid
        """

        if not self.initial_velocity:
            raise RuntimeError("Background velocity must be set before creating a grid.")
        
        self.grid_unc = unc_mag

        with open(self.__grid_path, 'r') as infile:
            lines = infile.readlines()

        lines[14] = f'{round(self.initial_velocity, 3):<24} c: Background velocity\n'

        if latgrid and longrid:

            lines[7] = f'{latgrid}                   c: Number of grid points in theta (N-S)\n'
            lines[8] = f'{longrid}                   c: Number of grid points in phi (E-W)\n'

        else:
            warnings.warn("latgrid and longrid not defined; leaving unchanged...")

        lines[9] = f'{self.region[0]}  {self.region[1]}          c: N-S range of grid (degrees)\n'
        lines[10] = f'{self.region[2]}  {self.region[3]}          c: E-W range of grid (degrees)\n'


        if noise and not noise_std:
            warnings.warn("No amplitude passed for noise - defaulting to 0.8!")


        if noise and not noise_seed:
            warnings.warn("No seed passed for noise - defaulting to 12324!")


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

        if checkerboard:
            lines[29] = '1                     c: Add checkerboard (0=no,1=yes)\n'
            lines[30] = f'{checker_val}                  c: Maximum perturbation of vertices\n'
            lines[31] = f'{checker_size}                     c: Checkerboard size (NxN)\n'
            if checker_spacing:
                lines[32] = '1                     c: Use spacing (0=no,1=yes)\n'
            else:
                lines[32] = '0                     c: Use spacing (0=no,1=yes)\n'
        else:
            lines[29] = '0                     c: Add checkerboard (0=no,1=yes)\n'

        if spike:
            lines[36] = '1                     c: Apply spikes (0=no,1=yes)\n'
        else:
            lines[36] = '0                     c: Apply spikes (0=no,1=yes)\n'

        with open(self.__grid_path, 'w') as outfile:
            outfile.writelines(lines)

    def create_grid(self,
                    copy_to_gridi: bool=True):
        
        """
        Creates a grid using grid2dss. Optionally, disables copying to gridi for manual handling.

        Parameters:
            copy_to_gridi (bool):   Whether to copy generated grid to gridi.vtx for FMST inversion
        """

        subprocess.run('grid2dss', cwd=self.__mkmodel_dir, shell=True, env=self.env)

        self.__grid_file_init = os.path.join(self.path, 'mkmodel', 'grid2d.vtx')

        _,self.grid_step, self.gridi = fmstUtils.read_grid_file(self.__grid_file_init)

        if copy_to_gridi:

            self.__gridi_file = os.path.join(self.path, 'gridi.vtx')

            shutil.copy(self.__grid_file_init, self.__gridi_file)

    def load_stations(self, station_path: str):

        """
        Loads stations for FMST from either an inventory xml or csv

        Accepts a csv with columns either of:
            ['network', 'station', 'lat', 'lon', 'elev']
            ['net','sta', 'Y', 'X', 'altitude'] (as generated by msnoise db dump)

        Parameters:
            station_path (str): Path to load
        """

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
            try:
                __stadf = pd.read_csv(station_path, usecols=__sta_cols)
            except:
                try: # msnoise db dump handling
                    __sta_cols_msn = ['net','sta', 'Y', 'X', 'altitude']
                    __stadf = pd.read_csv(station_path, usecols=__sta_cols_msn)
                    __stadf = __stadf.rename(columns={
                        'net': 'network',
                        'sta': 'station',
                        'Y': 'lat',
                        'X': 'lon',
                        'altitude': 'elev'
                    })
                    __stadf = __stadf[__sta_cols]

                except:
                    raise ValueError("Could not read station csv - check column names!")
            # Append rows to list from the CSV
            rows.extend(__stadf.values.tolist())
    
        else:
            raise ValueError("Supported formats are inventoryxml .xml and .csv!")
        
        # Convert the list of rows into a DataFrame
        __stadf = pd.DataFrame(rows, columns=__sta_cols)
        
        # Filter based on region
        if self.region:
            __stadf = __stadf[(__stadf['lon'] >= self.region[2]) & 
                            (__stadf['lon'] <= self.region[3]) &
                            (__stadf['lat'] >= self.region[1]) & 
                            (__stadf['lat'] <= self.region[0])]
        else:
            print("Region not set - skipping station crop...")
        
        # Assign to self.stations after filtering
        self.stations = __stadf
        self.station_count = int(len(self.stations))


    def load_velocity_pairs(self,
                            velocity_pairs_path: str,
                            phase_vel: float,
                            ignore_stations: list=None):

        """
        Loads velocity pair .json file as created by PyPhasePick

        Parameters:
            velocity_pairs_path (str): .json file of dispersion picks
            phase_vel (float):          specific velocity to load
        """

        genUtils.check_file_exists(velocity_pairs_path)

        __ext = os.path.splitext(velocity_pairs_path)[-1]

        if __ext != '.json':
            raise ValueError("Only json paired velocity inputs are supported")

        with open(velocity_pairs_path, 'r') as file:
            data = json.load(file)

        if ignore_stations:
            def keep_key(key):
                parts = key.split('_')
                return not any(sta in parts for sta in ignore_stations)

            data = {
                k: v for k, v in data.items()
                if keep_key(k)
            }

        self.velocity_pairs = {}

        for item, value in data.items():
            if phase_vel in value[0]:
                index_pf = value[0].index(phase_vel)
                self.velocity_pairs[item] = value[1][index_pf]

        if not self.velocity_pairs:
            raise ValueError(f"No velocity pairs found for frequency {phase_vel}")

    def read_station_pairs(self,
                           station_pairs_path: str,
                           drop: bool=False,
                           verbose: bool=False):
        
        """
        Reads station pair file
        Optionally drops stations from self.stations which are not found in the paired list.

        Parameters:
            station_pairs_path (str):   Path for csv of station pairs
                                        Format should be station1,station2,ZZ,TT,gcm,az,baz
                                        ZZ and TT values are boolean for whether such picks exist for the pair
            drop (bool):                Whether to drop unused stations from self.stations
            verbose (bool):             Prints information about dropped stations
        """

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
    
            if verbose:
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
                           arg: float=None,
                           verbose: bool=False):
        
        """
        Refines the station pair picks based on deviations from the mean velocity
        Saves the original unfiltered pairs to self.station_pairs_original
        
        Parameters:
            method (str):   'abs' or 'std'
                            'abs': removes velocities within $arg % percentile at each end of the distribution
                            'std': removes velocities $arg standard deviations from the mean
            arg (float):    argument for the method
            verbose (bool): prints out the number of discarded velocity pairs, if True
        """
        
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

        elif method == 'fit':

            with open(os.path.join(self.path, "rtravel.out"), "r") as file:
                ftimes = [list(map(float, line.strip().split())) for line in file]

            __num_paths = self.station_count ** 2

            trimmed_otimes = [o[:-1] for o in self.otimes]

            reltimes = [
                [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, ftime_row)]
                for otime_row, ftime_row in zip(trimmed_otimes, ftimes)
            ]

            collected_values = []

            for _, row in self.station_pairs_complete.iterrows():
                mat_idx = int(row['loc1'] * self.station_count + row['loc2'])
                collected_values.append(reltimes[mat_idx][-1])

            last_values_ordered = np.array(collected_values)

            __std_fraction = np.std([ k for k in last_values_ordered if k != 0 ])
            print(len([ k for k in last_values_ordered if k != 0 ]),
                  len([ k for k in trimmed_otimes if k[0] != 0 ]),
                      trimmed_otimes)

            if verbose:
                print('fit std:', __std_fraction)
                print('fit mean:', np.mean([ k for k in last_values_ordered if k != 0 ]))

            try:
                __sta_pairs = __sta_pairs[(last_values_ordered >= 1 - arg * __std_fraction) & (last_values_ordered <= 1 + arg * __std_fraction)]

            except ValueError as e:
                print("'fit' refinement requires the inversion to have already been performed with the same data")
                raise e

        else:
            raise ValueError("Supported methods of refine are 'abs' (absolute), 'std' (standard deviation) or 'fit' (relative fit)")

        if verbose == True:
            print("Discarded", __original_len - len(__sta_pairs), "velocity pairs with specified refine method (",__original_len , len(__sta_pairs),")")                

        __sta_pairs_clean = __sta_pairs.dropna(subset=['tt']).reset_index(drop=True)

        self.station_pairs_complete = __sta_pairs_clean

        self.station_pairs = __sta_pairs_clean[['loc1','loc2','tt']]

        self.refined = True

    def create_sources(self):

        """
        Creates sources.dat and receivers.dat required for FMST inversion
        Uses self.stations
        """

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
                      write: bool=True,
                      original: bool=False):
        
        """
        Creates otimes.dat as required for FMST inversion

        Parameters:
            unc (float):        Uncertainty parameter, in km/s, to write to file. If not set, defaults to 0.1 
            write (bool):       Whether to write out otimes. If not set, defaults to True
                                Can be used for non-destructive otime viewing
            original (bool):    Internal flag for refined data usage. If not set, defaults to False
        """
        
        __num_paths = self.station_count ** 2

        paths = [[0,0.,unc]] * __num_paths

        if original:
            for _, row in self.station_pairs_original[['loc1','loc2','tt']].iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['tt'],unc]
        else:
            for _, row in self.station_pairs.iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['tt'],unc]

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
                         If False, will rewrite onto existing files. Defaults to False
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
                            "smoothing":float,                  [eta]
                            "latitude_account":int,             [0=no, 1=yes]
                            "frac_G_size": float

            subiter (int): Value to write to `subiter.in` file.
            ttomoss (int): Value to write to `ttomoss.in` file. If not set, defaults to 6

        """

        if init:
            __config_files = ['fm2dss.in', 'ttomoss.in', 'misfitss.in', 'subinvss.in', 'subiter.in', 'residualss.in']

            for _ in __config_files:
                fmstUtils.create_file_from_template(os.path.join(self.path, _),self.templates_dir, _)

        if subiter:
            with open(os.path.join(self.path, 'subiter.in'), 'w') as file:
                file.write(str(subiter))

        if ttomoss:
            with open(os.path.join(self.path, 'ttomoss.in'), 'w') as file:
                file.write(str(ttomoss))
            self.iterations = ttomoss
        else:
            with open(os.path.join(self.path, 'ttomoss.in'), 'r') as file:
                self.iterations = int(file.read().strip()[0])

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
                   overwrite: bool=True,
                   iterate_paths: bool=False,
                   arg: float=2.0):
        
        """
        Runs the main inversion script ttomoss

        Parameters:
            verbose (bool):         If True, prints all terminal output. Defaults to False
            overwrite (bool):       If True: Overwrites self residuals and variance
                                    If False: Returns these instead in order:
                                        Original residual, variance; Final residual, variance
        """
            
        _ = subprocess.run('ttomoss', cwd=self.path, shell=True, check=True, capture_output=True, text=True, env=self.env)

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

        """
        Runs tslicess to produce an importable xyz layer

        Parameters:
            verbose(bool):  If True, prints out stdout and stderr. Defaults to False
        """

        if not os.path.exists(os.path.join(self.path, 'gmtplot')):
            os.makedirs(os.path.join(self.path, 'gmtplot'))

        if not os.path.exists(os.path.join(self.path, 'gmtplot', 'tslicess.in')):
            fmstUtils.create_file_from_template(os.path.join(self.path, 'gmtplot', 'tslicess.in'),self.templates_dir, 'tslicess.in')

        _ = subprocess.run('tslicess', cwd=os.path.join(self.path, 'gmtplot'), shell=True, check=True, capture_output=True, text=True, env=self.env)

        if verbose:
            print(_.stdout, _.stderr)

    def load_result_grid(self):

        """
        Reads the result grid generated after running tslicess
        """

        # read boundaries file generated by tslicess
        with open(os.path.join(self.path, 'gmtplot', 'bound.gmt')) as f:
            bounds = [float(x) for x in f]

        dx = bounds[4]
        dy = bounds[5]

        if dx < 1:
            x0, x1 = bounds[0], bounds[1]
            y0, y1 = bounds[2], bounds[3]
            base = [
                (np.arange(x0, x1, dx), np.arange(y0, y1, dy)),
                (np.arange(x0, x1 + dx, dx), np.arange(y0, y1,   dy)),
                (np.arange(x0, x1, dx),     np.arange(y0, y1+dy, dy)),
                (np.arange(x0, x1 + dx, dx), np.arange(y0, y1+dy, dy)),
            ]
        else:       # sometimes the boundaries file has strange values - manually setup the grid if so
            x0, x1 = self.region[2], self.region[3]
            y0, y1 = self.region[1], self.region[0]
            step_x = abs(x1 - x0) / dx
            step_y = abs(y1 - y0) / dy
            base = [
                (np.arange(x0, x1, step_x),     np.arange(y0, y1, step_y)),
                (np.arange(x0, x1+step_x, step_x), np.arange(y0, y1,   step_y)),
                (np.arange(x0, x1, step_x),     np.arange(y0, y1+step_y, step_y)),
                (np.arange(x0, x1+step_x, step_x), np.arange(y0, y1+step_y, step_y)),
            ]

        # read grid file generated by tslicess
        self.z_values = pd.read_csv(os.path.join(self.path,'gmtplot','grid2dv.z'),
                                    header=None).values.flatten()

        # permutate between off-by-one errors on dx and dy - can be either, probably a rounding error
        
        x_coords = None
        y_coords = None

        for x, y in base:
            if len(self.z_values) == len(x) * len(y):
                x_coords = x
                y_coords = y
                break

        if x_coords is None:
            raise ValueError("No grid dimension permutation matches Z length")
        
        # create a 2D grid for X, Y, and Z
        z_grid = self.z_values.reshape((len(x_coords), len(y_coords))).T

        # create a mesh grid for X and Y coordinates
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # stack X, Y, and Z grids into a single array for xyz2grd
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
                 label_stations: bool=False,
                 plot_caption: str=None,
                 save_fig: str=None,
                 show: bool=True):
        
        """
        Plots the final tomographic model onto a map

        Parameters:
            nlevels (int):          Number of levels for the colormap. If not set, defaults to 11
            cmap (str):             Colormap. If not set, defaults to "SCM/vik"
            reverse_cmap (bool):    If True, reverses the colormap. Defaults to True
            projection (str):       Projection parameter for gmt. If not set, defaults to 'M15c'
            plot_tomo (bool):       If True, plots the tomographic output. Defaults to True
            plot_rays (bool):       If True, plots the raypaths as found in rays.dat. Defaults to False
            plot_rays_v (bool):     If True, plots straight-line rays coloured by path velocity. Defaults to False
            plot_stations (bool):   If True, plots stations on the map as triangles. Defaults to False
            label_stations (bool):  If True, plots labels next to stations. Defaults to False
            plot_caption (str):     Add a caption to the figure in the lower left. Defaults to None
            save_fig (str):         If set, saves the figure to the given path. Defaults to None
            show (bool):            If True, shows the figure, else returns the figure. Defeaults to True
        """

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
                projection=projection,
                cmap= cpt,           
                interpolation="c",
                dpi=150
            )

        if plot_stations:
            receivers = pd.read_csv(os.path.join(self.path,'gmtplot','receivers.dat'), sep=r"\s+", header=None)
            receivers = receivers[[1, 0]] # lat and lon are wrong way around!


            fig.plot(
                data=receivers,
                region=gmt_region,                 # Map boundaries (equivalent to $bounds)
                projection=projection,               # Map projection (equivalent to $proj)
                style="t0.3c",
                fill="white",
                pen="black"
            )

            if label_stations:
                for _, station in self.stations.iterrows():  # Unpack the tuple
                    fig.text(
                        x=station.lon,  # Use attribute access instead of dict-style indexing
                        y=station.lat,
                        text=station.station,
                        pen="0.25p,black,solid",
                        fill="white",
                        font="10p",
                        offset="0.5/0.5"
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
                    pen=f"1p",
                    zvalue=row['vel'],
                    projection=projection,
                    cmap=True,
                    transparency=alpha  # Apply calculated alpha
                )

            fig.colorbar(cmap=True,
                frame=["x+lVelocity / km/s", "af"])
                
        if plot_caption:
            fig.text(
                text=plot_caption,
                position="LB",
                pen="0.25p,black,solid",
                fill="white",
                font="15p",
                offset="0.5/0.5"
            )

        if plot_tomo:
            fig.colorbar(cmap=cpt,
                frame=["x+lVelocity / km/s", "af"])
        
        if save_fig:
            fig.savefig(save_fig, crop=True)

        if show:
            fig.show()
        else:
            return fig

    def plot_hist(self,
                  use_model: bool=False,
                  mode: str='abs',
                  xlim: float=3,
                  save: str=None,
                  stat_lines: bool=False):

        if use_model:
            
            with open(os.path.join(self.path, "rtravel.out"), "r") as file:
                ftimes = [list(map(float, line.strip().split())) for line in file]

        __num_paths = self.station_count ** 2

        paths = [[0,0.]] * __num_paths

        if self.refined:
            for _, row in self.station_pairs_original.iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['gcm'] / self.initial_velocity]

        else:
            for _, row in self.station_pairs_complete.iterrows():
                idx = int(row['loc1'] * self.station_count + row['loc2'])
                paths[idx] = [1,row['gcm'] / self.initial_velocity]
        
        itimes = paths           

        old_otimes = self.otimes
        self.create_otimes(write=False, original=self.refined)
            
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
        plt.hist(last_values, bins=histbins, histtype='step', color='C0', edgecolor='red', lw=1, ls='--', label="original")
        plt.title("Travel Time Residuals")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xlim(xlims)

        if stat_lines:        
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

            plt.hist(last_values, bins=histbins, histtype='step', edgecolor='red', lw=1, ls="-", label="refined")

        if use_model:
            if mode=="abs":
                reltimes = [
                    [o - i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, ftime_row)]
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
            plt.hist(last_values, bins=histbins, histtype='step', edgecolor='black', lw=1, ls="-", label="final model")
            
        plt.legend()
        if save:
            plt.savefig(save)
        plt.show()

        self.otimes = old_otimes

    def lcurve(self,
               factor: str='smoothing',
               points: int=10,
               sample_range: list=[0.01, 100],
               iterations: int=3,
               logspace: bool=False,
               save: str=None):
        
        """
        Automatically performs l-curve analysis for a given inversion parameter and plots the result

        Parameters:
            factor (str):           'smoothing' or 'damping'; the factor to vary. Defaults to 'smoothing'
            points (int):           Number of points (= different values) to test. Deafults to 10
            sample_range (list):    Range over which to sample the given parameter. Defaults to [0.01,100]
            iterations (int):       Number of iterations to run the inversion for. Defaults to 3
                                        Note: This significantly impacts the time it takes to run!
            save (str):             Path to save the lcurve plot. Defaults to None.
        """


        if factor not in ['smoothing', 'damping']:
            raise ValueError("factor must be one of 'smoothing' or 'damping'")
        
        # copy out the existing output files before l-curve test

        files_to_backup = ["frechet.out", "gridc.vtx", "itimes.dat", "raypath.out", "residuals.dat", "rtravel.out", "subinvss.in", "subiter.in", "ttomoss.in"]
        file_paths = [os.path.join(self.path, f) for f in files_to_backup]
        temp_dir = fmstUtils.backup_files(file_paths)

        # create sample space defined by bounds and steps, logarithmically

        sample_space = [round(i, -int(np.floor(np.log10(abs(i))))) for i in np.logspace(np.log10(sample_range[0]), np.log10(sample_range[1]), points)]

        lcurve_dict = {}

        for s in tqdm(sample_space):
            self.config_ttomoss(subinvss={factor:s}, ttomoss=str(iterations))
            result_t = self.run_ttomoss(overwrite = False)

            result_m = subprocess.run('misfitss', cwd=self.path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)
                
            # failsafe check for good execution
            if result_m.returncode == 0:
                # regexs to extract variance and roughness from printed statistics
                variance_pattern = r"2 is\s*([0-9.E+-]+)"
                roughness_pattern = r"-1\) is\s*([0-9.E+-]+)$"
                
                # extract values from the output
                variance = float(re.search(variance_pattern, result_m.stdout).group(1))
                roughness = float(re.search(roughness_pattern, result_m.stdout).group(1))
            else:
                print(f"Error executing misfitss: {result_m.stderr}")

            if factor == 'smoothing':
                lcurve_dict[s] = [float(result_t[3]), roughness]

            else:
                lcurve_dict[s] = [float(result_t[3]), variance]

        if factor == 'smoothing':
            label_x = r"model roughness / (kms)$^{-1}$"

        else:
            label_x = r"model variance / (km/s)$^2$"
        
        fmstUtils.restore_files(temp_dir, [os.path.join(self.path, f) for f in files_to_backup])
        shutil.rmtree(temp_dir)  # clean up the temporary directory

        # plotting code
        pairs = sorted(lcurve_dict.items(), key=lambda kv: (kv[1][0], kv[1][1]))

        y = [v[0] for _, v in pairs]
        x = [v[1] for _, v in pairs]
        keys = [k for k, _ in pairs]

        fig, ax = plt.subplots()
        if logspace:
            ax.loglog(x, y, marker="o")
        else:
            ax.plot(x, y, marker="o")

        for i, key in enumerate(keys):
            ax.annotate(str(key), (x[i], y[i]), textcoords="offset points", xytext=(5,5))

        ax.set(ylabel=r'data variance / s$^2$', xlabel = label_x, 
               title=fr'{factor} factor varied')

        if save:
            plt.savefig(save)
        plt.show()

    def plot_vel_dots(self,
                     use_model: bool=True,
                     mode: str='abs',
                     std_dev: float=None):

        """

            Plots all velocity pairs as dots to show how the fit for each station pair changes between initial and final model

            By default, use_model is set to True to use the output final times from FMST. If False, only original dots are plotted.

            Plots lines between final and initial time deviation, with blue colour indicating improvements and red regressions

        """
            
        if use_model: 
            with open(os.path.join(self.path, "rtravel.out"), "r") as file:
                ftimes = [list(map(float, line.strip().split())) for line in file]

        else:
            ftimes = None

        __num_paths = self.station_count ** 2

        itimes = [[0,0.]] * __num_paths

        for _, row in self.station_pairs_complete.iterrows():
            idx = int(row['loc1'] * self.station_count + row['loc2'])
            itimes[idx] = [1,row['gcm'] / self.initial_velocity]

        trimmed_otimes = [o[:-1] for o in self.otimes]
        
        fig, ax = plt.subplots(figsize=(15,5))

        if mode=="abs":
            rtimes = [
                [o - i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, itime_row)]
                for otime_row, itime_row in zip(trimmed_otimes, itimes)
            ]
            reltimes = [
                [o - i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, ftime_row)]
                for otime_row, ftime_row in zip(trimmed_otimes, ftimes)
            ]

            ax.set_ylabel("Residual time / s")

        else:        
            rtimes = [
                [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, itime_row)]
                for otime_row, itime_row in zip(trimmed_otimes, itimes)
            ]
            reltimes = [
                [o / i if o != 0 and i != 0 else 0 for o, i in zip(otime_row, ftime_row)]
                for otime_row, ftime_row in zip(trimmed_otimes, ftimes)
            ]

            ax.set_ylabel("Residual / %")

        last_values = [row[-1] for row in rtimes if row[-1] != 0]
        ax.scatter(range(len(last_values)), last_values, color='grey', marker='x')

        mean = np.mean(last_values)
        stdev = np.std(last_values)
        ax.axhline(mean, color='grey', linestyle='dashed', linewidth=2, label=f'Mean')
        if std_dev:    
            ax.axhline(mean+2*stdev, color='black', linestyle='dashed', linewidth=2, label=f'2 StDev')
            ax.axhline(mean-2*stdev, color='black', linestyle='dashed', linewidth=2)
        ax.fill_between([0,len(last_values)],[mean+2*stdev]*2,[mean-2*stdev]*2, color='grey', alpha=0.1)
        ax.fill_between([0,len(last_values)],[mean+  stdev]*2,[mean-  stdev]*2, color='grey', alpha=0.3)

        last_values_f = [row[-1] for row in reltimes if row[-1] != 0]
        ax.scatter(range(len(last_values_f)), last_values_f, color='grey', marker='o')

        for idx, val in enumerate(last_values):
            if mode=='abs':
                i = 0
            else:
                i = 1
            
            if abs(i - last_values[idx]) >= abs(i - last_values_f[idx]):
                ax.plot([idx]*2,[last_values[idx],last_values_f[idx]],color="tab:blue")
            else:
                ax.plot([idx]*2,[last_values[idx],last_values_f[idx]],color="tab:red")

        plt.show()

    def plot_stations_vel(self, original:bool=False):
            
        if original and self.refined:
            flipped_df = self.station_pairs_original.rename(columns={'station1': 'station2', 'station2': 'station1'})
            combined_df = pd.concat([self.station_pairs_original, flipped_df], ignore_index=True)
            grouped = combined_df.groupby('station1')

        else:
            flipped_df = self.station_pairs_complete.rename(columns={'station1': 'station2', 'station2': 'station1'})
            combined_df = pd.concat([self.station_pairs_complete, flipped_df], ignore_index=True)
            grouped = combined_df.groupby('station1')
            
        
        boxplot_data = [group['vel'].values for _, group in grouped]

        # Plot boxplots
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(boxplot_data, vert=True, patch_artist=True, labels=grouped.groups.keys())
        ax.set_title('Boxplot of Velocities by Station')
        ax.set_xlabel('Station')
        ax.set_ylabel('Velocity')
        plt.xticks(rotation=90)
        
        plt.show()

    def checkerboard_test(self, checker_size: int=3, checker_val: float=0.3,checker_spacing: bool=False,
                           noise: float=None, noise_seed: int=12345, plot_input: bool=True, plot_output: bool=True):
        
        # copy out the existing output files before checkerboard test

        if plot_output:

            files_to_backup = ["frechet.out", "gridc.vtx", "itimes.dat", "raypath.out", "residuals.dat", "rtravel.out", "subinvss.in", "subiter.in", "ttomoss.in"]
            file_paths = [os.path.join(self.path, f) for f in files_to_backup]
            temp_dir = fmstUtils.backup_files(file_paths)

        # perform checkerboard test using the same paths as already defined

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.config_grid(checkerboard=True, checker_size=checker_size,
                          checker_val=checker_val, checker_spacing=checker_spacing)
        
        self.create_grid()

        src = os.path.join(self.path, "gridi.vtx")
        dst = os.path.join(self.path, "gridc.vtx")
        shutil.copy(src, dst)

        subprocess.run("fm2dss", cwd=self.path)

        if plot_input:
            self.run_tslicess()
            self.load_result_grid()
            self.plot_map(projection="M12c")

        with open(os.path.join(self.path, "rtravel.out"), "r") as file:
            lines = file.readlines()

        if noise:
            import random
            random.seed(noise_seed)
            new_lines = []
            for line in lines:
                parts = line.split()
                t = float(parts[-1])
                t = t + random.gauss(0.0, noise)
                base = " ".join(parts[:-1])
                new_lines.append(f"{base} {t:.6f}   {noise}\n")
        else:
            new_lines = [line.strip() + "   0.1\n" for line in lines]

        with open(os.path.join(self.path, "otimes.dat"), "w") as file:
            file.writelines(new_lines)

        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.config_grid(checkerboard=False)
        self.create_grid()

        self.config_ttomoss(init=True)
        self.run_ttomoss(overwrite=False)

        if plot_output:
            self.run_tslicess()
            self.load_result_grid()
            self.plot_map(projection="M12c")
        
            fmstUtils.restore_files(temp_dir, [os.path.join(self.path, f) for f in files_to_backup])
            shutil.rmtree(temp_dir)  # clean up the temporary directory
