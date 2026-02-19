from ngsolve import *
from netgen.occ import *
import numpy as np
import dill as pickle
from itertools import count
from safe_add_CFs import safe_add_CFs, interp_add_CFs
from scipy.sparse import lil_matrix

from os import listdir, makedirs
from os.path import isfile, join
from time import time
from datetime import timedelta

from create_observation import *

class grid_maker():
    def __init__(self, mmaker, ):

        self.mmaker = mmaker
        self.mesh = mmaker.mesh
        self.mid_radius = mmaker.mid_radius
        self.outer_radius = mmaker.outer_radius
        
        self.shape = shape

        self.integration_order = integration_order
        
        if sensor_radius is None:
            self.sensor_radius = sensor_radius
            sensor_radius_name = str(0)
        else:
            self.sensor_radius = sensor_radius
            sensor_radius_name = str(round(sensor_radius,5))
        self.base = 24
        
        self.cofes = mmaker.cofes
        
        self.complex_data = complex_data
        self.dtype = np.float32
            
        

        grid_dumps_directory = "grid_dumps/"
        makedirs(grid_dumps_directory, exist_ok = True)
        
        target_filename = "target_" + str(target_m) + "_shape_" + shape + "_sensorradius_" + sensor_radius_name + "_" + self.mmaker.target_filename
        self.target_filename = target_filename 
        
        self.load_flag = False
        # Attempt to find a stored grid with the same target
        for filename in listdir(grid_dumps_directory):
            if isfile(join(grid_dumps_directory, filename)):
                if target_filename in filename:
                    with open(grid_dumps_directory + filename, "rb") as input_file:
                        obj = pickle.load(input_file)

                        self.grid = obj["grid"]
                        self.sqrtm = obj["sqrtm"]
                        self.base = obj["base"]
                        self.Omat = obj["Omat"]
                        self.obstime = obj["obstime"]
                        self.circ_coords = obj["circ_coords"]
                        
                        self.load_flag = True
                        print("Loaded stored grid " + filename + "...")
                        break

        if not self.load_flag:
            print("No saved grid found, generating...")
            filename = grid_dumps_directory + target_filename
            
            scatterers = self.mmaker.scatterers
            
            # Create observation grid
            
           
            
        # Create sparse observation matrix for efficiency.

    def finalise_grid(self):
        if self.load_flag:
            return
            
        cofes = self.mmaker.cofes
        
        
        obj = {"grid": self.grid, "sqrtm": self.sqrtm, "base": self.base, "Omat": self.Omat, \
               "circ_coords": self.circ_coords, "obstime": self.obstime}
        with open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)    
        print("Successfully stored mesh, grid & observation functions with m_sensors = " + str(self.m_sensors) + ".")