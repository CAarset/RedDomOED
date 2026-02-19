import numpy as np
from ngsolve import TaskManager, GridFunction, IfPos

def pairg(myg):
    for i in myg:
        try:
            yield i + next(myg)
        except StopIteration:
            yield i

def comp(myg, length):
    for _ in range(int(np.ceil(np.log2(length)))):
        myg = pairg(myg)
    for i in myg:
        return i
    
def safe_add_CFs(CFs, weights = None, length = None, supported = False):
    with TaskManager():
        if weights is not None:
            def wCFs():
                for tup, weight in zip(CFs, weights):
                    if supported:
                        cf, supp = tup
                    else:
                        cf = tup
                        supp = tup
                    yield IfPos(supp, cf * weight, 0)
                    #else:
                    #    cf = tup
                    #    yield cf * weight
            try:
                length = len(weights)
            except TypeError:
                pass
    return comp(wCFs(), length = length)

def interp_add_CFs(CFs, fes, weights = None):
    gf = GridFunction(fes)
    gf0 = GridFunction(fes)
    if weights is None:
        with TaskManager():
            for cf in CFs:
                gf0.Set(cf)
                gf.vec.data += gf0.vec.data
    else:
        with TaskManager():
            for cf, weight in zip(CFs, weights):
                if weight != 0:
                    gf0.Set(weight * cf)
                    gf.vec.data += gf0.vec.data
    return gf