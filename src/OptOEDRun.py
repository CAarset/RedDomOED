# Computes 

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

R = RDOED.R
Q = RDOED.Q
A = mmaker.A
AT = mmaker.AT
C = mmaker.C
O = gmaker.O

mdigits = int(np.log10(m)+1)

output_path = "opt_outputs"
makedirs(output_path, exist_ok = True)

M = mmaker.M
dfes = mmaker.designfes
dmesh = mmaker.designmesh

np.seterr(divide='ignore')
targets = np.arange(1,m_sensors)

fn = RDOED.output_filename
try:
    with open(fn, "rb") as filename:
        obj = pickle.load(filename)
    print("Successfully loaded previous results.")
except:
    obj = {}
    
    obj["w0s"] = {}
    obj["ws"] = {}
    obj["wseqs"] = {}
    obj["ps"] = {}
    obj["times"] = {}
    
w1 = None

for target in targets:
    
    # Try to load optimal design from file
    if target in obj["ws"].keys():
        print("Skipping ",target,"...",sep="")
        w1 = obj["w1s"][target]
        continue
    
    start = time()
    RDOED.Opt(target = target, w1 = w1, verbose = True)
    
    #w0 = deepcopy(RDOED.design)
    w1 = deepcopy(RDOED.global_design)
    
    obj["w1s"][target] = deepcopy(w1)
    obj["ws"][target] = deepcopy(RDOED.design)
    obj["wseqs"][target] = deepcopy(RDOED.design_sequence)
    try:
        obj["ps"][target] = deepcopy(RDOED.ps)
    except:
        obj["ps"][target] = None
    obj["times"][target] = time() - start

    with open(fn, "wb") as filename:
        pickle.dump(obj, filename)
