from OEDAlgorithm import *
from scipy.optimize import minimize, LinearConstraint, Bounds
from copy import deepcopy
from scipy.special import comb
from itertools import combinations

class RNGOED(OEDAlgorithm):
    def __init__(self, tries = 1e3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tries = int(tries)
        self.allvals = []
        
    def RNG(self, target, tries = None, subset = None, verbose = False):
        
        self.verbose = verbose
        self.allvals = []
        
        if tries is None:
            tries = self.tries
        tries = int(tries)
        
        wRNG = np.zeros(self.m_sensors)
        valRNG = np.inf

        potential = comb(N = self.m_sensors, k = target, exact = True)
        candidates = np.arange(self.m_sensors)

        if potential < tries:
            for i, inds in enumerate(combinations(candidates,target)):
                inds = np.array(inds, dtype = int).ravel()
                
                if self.verbose:
                    print("\r","Target: ",target,", value: ",valRNG,", try ", i,"/",potential,", inds: ",inds,sep="",end="")
                
                w = np.zeros(self.m_sensors)
                w[inds] = 1
                val = self.J_init.eval(w)
                self.allvals.append(val)
                
                if val < valRNG:
                    valRNG = val
                    self.design = deepcopy(w)
        else:
            for i in range(tries):
                if subset is not None:
                    inds = self.rng.choice(subset, target, replace = False)
                else:
                    inds = self.rng.choice(self.m_sensors, target, replace = False)

                if self.verbose:
                    print("\r","Target: ",target,", value: ",valRNG,", try ", i,"/",tries,", inds: ",inds,sep="",end="")
                w = np.zeros(self.m_sensors)
                w[inds] = 1
                val = self.J_init.eval(w)
                self.allvals.append(val)
                
                if val < valRNG:
                    valRNG = val
                    self.design = deepcopy(w)
                    
    def CircRNG(self, target, tries = None, verbose = False):
        
        self.verbose = verbose
        self.allvals = []
        
        if tries is None:
            tries = self.tries
        tries = int(tries)
        
        base = int(self.gmaker.base)
        circs = int(np.ceil(np.log2(target / base + 1)))
        inners = base * int(np.ceil(2**(circs-1)-1))
        
        w0 = np.zeros(self.m_sensors)
        w0[:inners] = 1
        #w = deepcopy(w0)
        
        reduced_target = target - inners
        potential = comb(N = int(base * 2**(circs-1)), k = reduced_target, exact = True)
        candidates = inners + np.arange(int(base * 2**(circs-1)))
        
        if self.verbose:
            print("\nCircs:",circs)
            print("Inners:",inners)
            print("Inner indices:",np.argwhere(w0).ravel())
            print("Potential:",potential)
            print("Candidates:",candidates)
        
        valRNG = np.inf
        if potential < tries:
            for i, inds in enumerate(combinations(candidates,reduced_target)):
                inds = np.array(inds, dtype = int).ravel()
                
                if self.verbose:
                    print("\r","Target: ",target,", reduced target: ", reduced_target,", value: ",valRNG,", try ", i,"/",potential,", inds: ",inds,", current sum: ",np.sum(w),sep="",end="")
                
                w = deepcopy(w0)
                w[inds] = 1
                val = self.J_init.eval(w)
                self.allvals.append(val)
                
                if val < valRNG:
                    valRNG = val
                    self.design = deepcopy(w)

        else:
            for i in range(tries):
                inds = self.rng.choice(candidates, reduced_target, replace = False)
                
                if self.verbose:
                    print("\r","Target: ",target,", reduced target: ", reduced_target,", value: ",valRNG,", random try ",i,"/",tries,", inds: ",inds,", current sum: ",np.sum(w),sep="",end="")
                
                w = deepcopy(w0)
                w[inds] = 1
                val = self.J_init.eval(w)
                self.allvals.append(val)
                
                if val < valRNG:
                    valRNG = val
                    self.design = deepcopy(w)