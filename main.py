import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from obspy import read_inventory
import pygmt

from fmstUtils import fmstUtils, genUtils

class fmst:

    def __init__(self,
                 path: str):
        """
        Initiliazes the fmst class

        Requires a file path for operations and files to be built within
        """

        if not path:
            raise ValueError("The path must be provided.")

        genUtils.check_file_exists(path)

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

            fmstUtils.create_file_from_template(self.__grid_path, 'grid2dss.in')

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
                    noise_std: float=0.8,
                    noise_seed: int=12324,
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

        lines[9] = f'{region[0]}  {region[1]}          c: N-S range of grid (degrees)\n'
        lines[10] = f'{region[2]}  {region[3]}          c: E-W range of grid (degrees)\n'


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

        self.region = region

    def create_grid(self,
                    copy_to_gridi: bool=True):

        subprocess.run('grid2dss', cwd=self.__mkmodel_dir, shell=True)

        self.__grid_file_init = os.path.join(self.path, 'mkmodel', 'grid2d.vtx')

        _,self.grid_step, self.gridi = fmstUtils.read_grid_file(self.__grid_file_init)

        if copy_to_gridi:

            self.__gridi_file = os.path.join(self.path, 'gridi.vtx')

            shutil.copy(self.__grid_file_init, self.__gridi_file)

    def load_stations(self,
                      station_path: str):

        genUtils.check_file_exists(station_path)
        
        __ext = os.path.splitext(station_path)
        __sta_cols = ['network','station','lat','lon','elev']
        
        if __ext == '.xml':
            __inv = read_inventory(station_path)

            for network in inv:
                for station in network:
                    __stadf = pd.concat([__stadf, pd.DataFrame(
                        [[network.code,
                        station.code,
                        station.latitude,
                        station.longitude,
                        station.elevation]], columns=__sta_cols
                    )], ignore_index=True)

        elif __ext == '.csv':
            __stadf == pd.read_csv(station_path, usecols=__sta_cols)

        else:
            raise ValueError("Supported formats are inventoryxml .xml and .csv!")
                    
        __stadf = __stadf[(__stadf['lon'] >= self.region[2]) & (__stadf['lon'] <= self.region[3]) &
                         (__stadf['lat'] >= self.region[1]) & (__stadf['lat'] <= self.region[0])]

        self.stations = __stadf

    def load_velocity_pairs(self,
                            velocity_pairs: str):
        
        genUtils.check_file_exists(velocity_pairs)
        
        __ext = os.path.splitext(velocity_pairs)
        
        if __ext != '.json':
            raise ValueError("Only json paired velocity inputs are supported!")

        with open(phase_vel_path, 'r') as file:
            data = json.load(file)
        
        self.velocity_pairs = {}
        
        for item, value in data.items():
            if phase_vel in value[0]:
                index_pf = value[0].index(phase_vel)
                self.velocity_pairs[item] = value[1][index_pf]

    def read_station_pairs(self,
                           station_pairs_path: str,
                           refine: bool=False,
                           refine_method: str=None,
                           refine_arg: float=None):

        __sta_pairs = pd.read_csv(station_pairs_path)
    
        __sta_pairs['vel'] = None
        
        for item, value in vels.items():
            sta1 = item.split("_")[1]
            sta2 = item.split("_")[-1]
        
            __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'vel'] = value
            __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'loc1'] = int(self.stations.index[self.stations['station'] == sta1].tolist()[0]) 
            __sta_pairs.loc[(__sta_pairs['station1'] == sta1) & (__sta_pairs['station2'] == sta2), 'loc2'] = int(self.stations.index[self.stations['station'] == sta2].tolist()[0])
        
        __sta_pairs['gcm'] /= 1000 # converts from metres to kilometres
        
        __sta_pairs['tt'] = __sta_pairs['gcm'] / __sta_pairs['vel']

        __original_len = len(__sta_pairs)
    
        if refine:
            if refine_method == 'std':
                __mean_vel = np.mean(__sta_pairs['vel'])
                __std_vel = np.std(__sta_pairs['vel'])
                
                __sta_pairs = __sta_pairs[(__sta_pairs['vel'] >= mean_vel - refine_arg * std_vel) & (__sta_pairs['vel'] <= mean_vel + refine_arg * std_vel)]

            elif refine_method == 'abs':
                __lower_bound = np.percentile(__sta_pairs['vel'], refine_arg)
                __upper_bound = np.percentile(__sta_pairs['vel'], 100 - refine_arg)
                
                __sta_pairs = __sta_pairs[(__sta_pairs['vel'] >= __lower_bound) & (__sta_pairs['vel'] <= __upper_bound)]

            print("Discarded", __original_len - len(__sta_pairs), "velocity pairs with specified refine method")                

        else:
            __sta_pairs_clean = __sta_pairs[['loc1','loc2','tt']].dropna(subset='tt')

        self.station_pairs = __sta_pairs_clean

    def remove_unused_stations(self):

        __used_stations = list(pd.unique(self.station_pairs[['col1', 'col2']].values.ravel()))

        __orig_stations = len(self.stations)

        self.stations = self.stations[self.stations['sta'].isin(__used_stations)]

        print("Removed", __orig_stations - len(self.stations), "unused stations!")

    def create_sources(self):

        __sources = self.stations[['lat','lon']]

        self.station_count = int(len(__sources))

        sdf.to_csv(os.path.join(self.path,'sources.dat'), sep=r" ", header=None, index=False)
        
        with open(os.path.join(self.path,'sources.dat'), 'r') as file:
            lines = file.readlines()
        
        lines.insert(0, f"   {self.station_count}\n")

        with open(os.path.join(self.path,'sources.dat'), 'w') as file:
            file.writelines(lines)
        
        sdf.to_csv(os.path.join(self.path,'receivers.dat'), sep=r" ", header=None, index=False)
        
        with open(os.path.join(self.path,'receivers.dat'), 'r') as file:
            lines = file.readlines()
        
        lines.insert(0, f"   {self.station_count}\n")
        
        with open(os.path.join(self.path,'receivers.dat'), 'w') as file:
            file.writelines(lines)

    def create_otimes(self,
                      unc: float=0.1):
        
        __num_paths = self.station_count ** 2

        paths = [[0,0.,unc]] * __num_paths
    
        for _, row in self.station_pairs.iterrows():
            idx = int(row['loc1'] * self.station_count + row['loc2'])
            paths[idx] = [1,row['tt'],0.1]
        
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
                            "smoothing":int,                    [eta]
                            "latitude_account":int,             [0=no, 1=yes]
                            "frac_G_size": float
                             
            subiter (int): Value to write to `subiter.in` file.
            ttomoss (int): Value to write to `ttomoss.in` file.

        """

        if init:
            __config_files = ['fm2dss.in', 'ttomoss.in', 'misfitss.in', 'subinvss.in', 'subiter.in', 'residualss.in']
            
            for _ in __config_files:
                fmstUtils.create_file_from_template(os.path.join(self.path, _), _)

        if subiter:
            with open(os.path.join(self.path, 'subiter.in'), 'w') as file:
                file.write(subiter)
        
        if ttomoss:
            with open(os.path.join(self.path, 'ttomoss.in'), 'w') as file:
                file.write(ttomoss)

        if subinvss:
            __params = {'damping':float, 'subspace_dimension':int, '2nd_derivative_smoothing':int,
                       'smoothing':int, 'latitude_account':int, 'frac_G_size': float}

            fmstUtils.process_file(os.path.join(self.path, 'subinvss.in'), 11, __params, subinvss)

        if misfitss:
            __params = {'dicing': (int, int), 'earth_radius':float}

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
                   verbose: bool=False):

        _ = subprocess.run('ttomoss', cwd=self.path, shell=True, check=True, capture_output=True, text=True)

        if verbose:
            print(_.stdout)

    def run_tslicess(self, verbose: bool=False):

        if not os.path.exists(os.path.join(self.path, 'gmtplot')):
            os.makedirs(os.path.join(self.path, 'gmtplot'))

        if not os.path.exists(os.path.join(self.path, 'gmtplot', 'tslicess.in')):
            fmstUtils.create_file_from_template(os.path.join(self.path, 'gmtplot', 'tslicess.in'), 'tslicess.in')

        _ = subprocess.run('tslicess', cwd=os.path.join(self.path, 'gmtplot', shell=True, check=True, capture_output=True, text=True)

       if verbose:
           print(_.stdout)

   def load_result_grid(self):

        with open(os.path.join(self.path, 'gmtplot', 'bound.gmt'), "r") as file:
            bounds = [float(line.strip()) for line in file]
    
        if bounds[4] < 1:   
            x_coords = np.arange(bounds[0], bounds[1], bounds[4])
            y_coords = np.arange(bounds[2], bounds[3] + bounds[5], bounds[5])
    
        else:
            x_coords = np.arange(region[0], region[1], abs(region[0]-region[1])/bounds[4])
            y_coords = np.arange(region[2], region[3], abs(region[2]-region[3])/bounds[5])

        self.z_values = pd.read_csv(f"{ttomoss_path}gmtplot/grid2dv.z", header=None).values.flatten()

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

    def map_tomo(self,
                 nlevels: int=11,
                 cmap: str="SCM/vik",
                 reverse_cmap: bool=True,
                 projection: str='M15c',
                 plot_rays: bool=False,
                 plot_stations: bool=False,
                 plot_caption: bool=False
                 ):

        orig_grid = pygmt.xyz2grd(data=self.xyz_data,
                                region=self.region,
                                spacing=[self.bounds[6], self.bounds[7]],
                                registration="pixel")
        
        fig = pygmt.Figure()
        
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
            region=region,
            projection=projection,  # Mercator projection (6 inches wide)
            cmap= cpt,           # f"{ttomoss_path}gmtplot/velgradabs1.cpt",  # Color palette
            frame=True,  # Frame with ticks
            interpolation="c",
            dpi=150
        )

        if plot_stations:
            receivers = pd.read_csv(f"{ttomoss_path}gmtplot/receivers.dat", sep="\s+", header=None)
            receivers = receivers[[1, 0]] # lat and lon are wrong way around!
    
    
            fig.plot(
                data=receivers,          
                region=region,                 # Map boundaries (equivalent to $bounds)
                projection=projection,               # Map projection (equivalent to $proj)
                style="t0.3c",
                fill="white",
                pen="black"
            )
        
        # fig.basemap(region=region, projection="M6i", frame=["a10f5", "a10f5"])  # Frame with intervals
        fig.coast(
            shorelines=True,  # Draw coastlines
            borders=[1, 2],  # Show internal administrative boundaries and countries
            resolution="f",   # Full resolution coastline
        )
        
        fig.colorbar(cmap=cpt,
            frame=["x+lVelocity / km/s", "af"])

        if plot_rays:
            fig.plot(
                data=f"{ttomoss_path}gmtplot/rays.dat",          # Input data file
                region=region,            # Map boundaries (equivalent to $bounds)
                projection=projection,          # Map projection (equivalent to $proj)
                pen="0.5p",               # Line width of 0.5 points (equivalent to -W0.5)
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

        