import numpy as np
import scipy
#import zfista

from copy import deepcopy
from time import time
from datetime import timedelta
from copy import deepcopy

from ngs_grid import *
from ngs_mesh import *
from RedDomOED import *
from GreedyOED import *
from FISTAOED import *
from OptOED import *
from RNGOED import *

import random
random.seed(19)
rng = np.random.default_rng(19)

dim = 2
scatterers = 3
order = 2

import sys

try:
    target_m = int(sys.argv[1])
except:
    target_m = int(3e2)
    print("Running with default target m =",target_m,"for testing purposes.")
#try:
#    do_RedDom = bool(sys.argv[2])
#except:

OED_types = ["Opt", "RedDom", "Greedy", "FISTA"]
OED_type = OED_types[-1]

noise_level = 1e-2 # This is relative noise level, in percent rel. to operator norm $FC$. Currently assuming identity noise cov (white noise).
tol = 1e-15

maxh = 0.03

#omegas = 50
#omegas = 40
#omegas = [30, 35, 40, 45, 50]
omegas = [20, 25, 30, 35, 40, 45, 50]

P = 2 # Stepping size for 0-norm approximation

#mode = "qr"
#mode = "auto"
mode = "rsi"


#shape = "gauss"
#shape = "round"

if 'sensor_radius' not in locals():
    shape = "pointwise"
    sensor_radius = None
else:
    if sensor_radius is None:
        shape = "pointwise"
    else:
        shape = "gauss"

PML = False
refine_inner = 0
clear = False
if clear:
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

mmaker = mesh_maker(maxh = maxh, scatterers = scatterers, \
                     order = order, PML = PML, refine_inner = refine_inner)
mesh = mmaker.mesh

if mode == "qr":
    target_rank = None

gmaker = grid_maker(mmaker, target_m = target_m, shape = shape, sensor_radius = sensor_radius)

def target_rank(m,n):
    tr = max(min(m,150),\
                3*max(\
                    int(10*float(m)**(1/4)),\
                    int(5*float(m)**(1/3))\
                ))
    #target_rank = 233
    #target_rank = 100
    #tr = 200
    return min(min(tr, m), n)

#target_rank = min(target_rank,63)#min(target_rank,40)
start = time()

kwargs = { \
          "mmaker": mmaker, "gmaker": gmaker,\
          "omegas": omegas, \
          "noise_level": noise_level, "tol": tol, "target_rank": target_rank, \
          "shape": shape, "mode": mode, "max_iters": 1e5, "P": P \
         }

if 0:
    if OED_type.casefold() == "Opt".casefold():
        RDOED = OptOED(**kwargs)
    elif OED_type.casefold() == "RedDom".casefold():
        RDOED = RedDomOED(**kwargs)
    elif OED_type.casefold() == "Greedy1st".casefold():
        RDOED = Greedy1stOED(**kwargs)
    elif OED_type.casefold() == "FISTA".casefold():
        RDOED = FISTAOED(**kwargs)
    else:
        raise ValueError("Argument OED_type must be one of Opt, RedDom, Greedy and FISTA (case insensitive)!")

FISTAOED = FISTAOED(**kwargs)
OptOED = OptOED(**kwargs)
GreedyOED = GreedyOED(**kwargs)
RNGOED = RNGOED(verbose = False, **kwargs)
RDOED = OptOED #FISTAOED

output_path = "opt_outputs"
try:
    mkdir(output_path)
except:
    pass
fn = output_path + "/" + RDOED.solver + "_peps_" + str(RDOED.peps) + "_" + RDOED.target_filename
RDOED.output_filename = fn
    
stop = time()

print("Red-Dom setup in " + str(timedelta(seconds=stop-start)) + "...")
