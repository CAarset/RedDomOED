# Computes p-relaxed optimal binary designs for every sensor target (slow)

from runOED import *
from os import makedirs
from copy import deepcopy
import dill as pickle
from util import *
from time import time

n, m, m_sensors, m_obs = RDOED.n, RDOED.m, RDOED.m_sensors, RDOED.m_obs
print("m = ",m,", n = ",mmaker.n,", ell = ",RDOED.ell,sep="")

mesh = mmaker.mesh
fes = mmaker.fes
print(m,mmaker.n,RDOED.ell)

mdigits = int(np.log10(m)+1)

output_path = "opt_outputs"
makedirs(output_path, exist_ok = True)

M = mmaker.M
dfes = mmaker.designfes
dmesh = mmaker.designmesh

np.seterr(divide='ignore')
targets = np.arange(1,m_sensors)

fn = RDOED.output_filename + "_RNG"
try:
    with open(fn, "rb") as filename:
        obj = pickle.load(filename)
    print("Successfully loaded previous results.")
except:
    obj = {}
    
    obj["wRNGs"] = {}
    obj["allvals"] = {}
    obj["times"] = {}
    
for target in targets:
    
    # Try to load optimal design from file
    if target in obj["wRNGs"].keys():
        print("Skipping ",target,"...",sep="")
        continue

    start = time()
    RNGOED.RNG(target = target, tries = int(1e3), verbose = True)
    
    obj["wRNGs"][target] = RNGOED.design
    obj["allvals"][target] = RNGOED.allvals
    obj["times"][target] = time() - start

    with open(fn, "wb") as filename:
        pickle.dump(obj, filename)
