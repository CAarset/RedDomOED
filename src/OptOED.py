from OEDAlgorithm import *
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds
#from cyipopt import minimize_ipopt
from FISTA import FISTA
#from jax import jit, grad, jacfwd, jacrev
from copy import deepcopy

class OptOED(OEDAlgorithm):
    def __init__(self, peps = 1e-1, min_p = 1e-2, solver = 'SLSQP', *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.max_p_steps = int(max_p_steps)
        #self.max_p_steps_digits = int(np.log10(self.max_p_steps)+1)
        self.min_p = min_p
        self.global_design = None
        self.peps = peps
        self.solver = solver
        
    def Opt(self, target, w1 = None, verbose = False):

        p = 1
        p_old = 1
        peps = deepcopy(self.peps)
        min_p = self.min_p
        #max_p_steps = deepcopy(self.max_p_steps)

        self.verbose = verbose

        # Initial solve for p = 1
        self.dom_indices = np.array([],dtype=int)
        self.red_indices = np.array([],dtype=int)
        self.free_indices = np.arange(self.m_sensors)
        self.update()
    
        J = self.J
        eva = lambda w: J.eval(w)
        jac = lambda w: J.jac(w)
        
        ms = self.m_sensors
        cns = (LinearConstraint(A = np.eye(ms), lb = 0, ub = 1), LinearConstraint(A = np.ones(ms), ub = target))

        opts = {"maxiter": self.max_iters}
        kwargs = {"tol": self.tol, "options": opts, "jac": jac, "constraints": cns}

        w = w1
        if w is None:
            w = np.ones(ms) * target / ms

        print("Initial solve for target = ",target,"...",sep="")
        res = minimize(fun = eva, x0 = w, method='SLSQP', **kwargs)
        if not res["success"]:
            print("\n","-"*50)
            print(res)
        w = res["x"]

        # Set binary indices and store initial red/dom/frees
        w = self.set_binary(w = w, target = target, set_red_dom = True)
        self.global_design = w
        
        initial_dom_indices = deepcopy(self.dom_indices)
        initial_red_indices = deepcopy(self.red_indices)
        initial_free_indices = deepcopy(self.free_indices)

        w = w[self.free_indices]
        initial_w = deepcopy(w)
        # Increase ftol for non-convex solves
        opts["ftol"] = 1e-10
    
        refines = 0
        while True:
            self.dom_indices = deepcopy(initial_dom_indices)
            self.red_indices = deepcopy(initial_red_indices)
            self.free_indices = deepcopy(initial_free_indices)
            self.update()

            w = deepcopy(initial_w)
    
            self.design_sequence = []
    
            self.ps = []
            
            while p > min_p:
                p_old = deepcopy(p)
                p *= 1 - peps
                self.ps.append(p)
                ip = 1/p
                if self.verbose:
                    print("\r","Opt OED for target = ",target,", p = ","{:.2e}".format(p), \
                          ", doms: ",self.dom_indices.size, \
                          ", reds: ",self.red_indices.size, \
                          ", free: ",self.free_indices.size,sep="",end="")
                    
                def wip(w,t=0):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        out = w**(ip-t)
                    out[np.isnan(out) | np.isinf(out)] = 0
                    out = np.fmin(out,1)
                    return out
            
                def wp(w):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        out = w**p
                    out[np.isnan(out) | np.isinf(out)] = 0
                    out = np.fmin(out,1)
                    return out
    
                J = self.J
                eva = lambda w: J.eval(wip(w))
                jac = lambda w: J.jac(wip(w)) * ip * wip(w,1)
                #hessp = lambda w, v: ip **2 * wip(w,1) * J.hess_matvec(wip(w,0),wip(w,1) * v) \
                #                   - ip * (ip - 1) * J.jac(wip(w,0)) * wip(w,2) * v
            
                msc = len(w)
                cns = (LinearConstraint(A = np.eye(msc), lb = 0, ub = 1), LinearConstraint(A = np.ones(msc), ub = target - self.dom_indices.size))
                
                kwargs["jac"] = jac
                kwargs["constraints"] = cns
                
                res = minimize(fun = eva, x0 = w, method='SLSQP', **kwargs) # ** (p/p_old)
                w = res["x"]
                
                vanishing = wip(w) < self.tol
            
                if vanishing.any():
                    w[vanishing] = 0
            
                    w = self.insert(w)
                    self.red_indices = np.argwhere(w == 0).ravel()
                    self.update()
                    w = w[self.free_indices]
            
                wi = self.insert(w)
                self.design_sequence.append(wi)
                
                if not res["success"]:
                    print("\n","-"*50)
                    print(res)

                active = self.free_indices.size + self.dom_indices.size
                if active == target:
                    wipwi = np.zeros(ms)
                    wipwi[self.free_indices] = 1
                    wipwi[self.dom_indices] = 1
                    print("Exactly",target,"indices remaining, finalising...")
                elif active < target:
                    print("Lost too many indices...")
                    break
                else:
                    wipwi = wip(wi)
                
                if self.nonb(wipwi) < self.tol and np.sum(wipwi) <= target + self.tol:
                    self.vprint("\nFound binary design for target " + str(target) + \
                                " with objective value " + str(self.J_init.eval(wipwi)) + ", terminating...")
                    self.vprint("-"*50)
                    self.design = wipwi
                    return

            self.vprint("\nNo binary design found for target " + str(target) + ", refining...")
            self.design = wipwi
            p = 1
            p_old = 1
            peps /= 2 #return
            min_p /= 2
            #max_p_steps *= 10
            refines += 1
            if refines >= 5:
                self.vprint("\nRefinement failed for target " + str(target) + ", breaking...")
                self.vprint("-"*50)
                return
