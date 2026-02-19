import numpy as np

from copy import deepcopy
from time import time
from datetime import timedelta
from copy import deepcopy

import random

import os

from ngs_mesh import mesh_maker
from OptOED import OptOED

from create_paper_geometry import create_paper_geometry

def runOED(maxh = 0.03, order = 1,
           refine_inner = 1, post_refine = 1, 
           noise_level = 1e-2, noise_type = "continuous",
           target_m = int(3e2), sensor_radius = 0, sensor_shape = "round", verbose = False, integration_order = 5, clear = False, force_clear = False):

    # Noise level is relative noise level, in percent rel. to operator norm $FC$. Currently assuming identity noise cov (white noise).
    
    random.seed(19)
    rng = np.random.default_rng(19)
    
    OED_types = ["Opt", "RedDom", "Greedy", "FISTA"]
    OED_type = OED_types[-1]
    
    tol = 1e-15
    
    #omegas = 50
    #omegas = 40
    #omegas = [30, 35, 40, 45, 50]
    omegas = [20, 25, 30, 35, 40, 45, 50]
    
    P = 2 # Stepping size for 0-norm approximation
    
    mode = "rsi"
    
    if sensor_radius == 0:
        sensor_shape = "pointwise"
    
    if clear or force_clear:
        if force_clear:
            answer = "yes"
            force_clear = False
            print("Warning: Forcibly clearing existing.")
        else:
            answer = input("Delete old storage?")
        if answer.lower() in ["y","yes"]:
            for mydir in ["decomps","grid_dumps","ngs_dumps","opt_outputs"]:
                try:
                    for f in os.listdir(mydir):
                        os.remove(os.path.join(mydir, f))
                except:
                    pass
        elif answer.lower() in ["n","no"]:
            print("Skipping...")
        else:
            assert "Input should be y/n!"
    
    geo_creator = create_paper_geometry()
    
    mmaker = mesh_maker(rng = rng, geo_creator = geo_creator,
                        maxh = maxh,
                        order = order, refine_inner = refine_inner, post_refine = post_refine,
                        target_m = target_m, 
                        sensor_shape = sensor_shape, sensor_radius = sensor_radius, 
                        integration_order = integration_order)
    
    if mode == "qr":
        target_rank = None
    else:
        def target_rank(m,n):
            tr = max(min(m,150),\
                        3*max(\
                            int(10*float(m)**(1/4)),\
                            int(5*float(m)**(1/3))\
                        ))
            return min(min(2*tr, m), n)
    
    start = time()
    
    kwargs = { \
              "mmaker": mmaker,
              "omegas": omegas,
              "noise_level": noise_level, "noise_type": noise_type,
              "tol": tol, "target_rank": target_rank,
              "sensor_shape": sensor_shape, "mode": mode, "max_iters": 1e5, "P": P,
              "verbose": verbose
             }
    
    RDOED = OptOED(**kwargs)
    
    output_path = "opt_outputs"
    try:
        mkdir(output_path)
    except:
        pass
    fn = output_path + "/" + RDOED.solver + "_peps_" + str(RDOED.peps) + "_" + RDOED.target_filename
    RDOED.output_filename = fn
        
    stop = time()
    
    print("Red-Dom setup in " + str(timedelta(seconds=stop-start)) + "...")
    return RDOED