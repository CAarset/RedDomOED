from OEDAlgorithm import *
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds, bisect
from FISTA import FISTA
import juliacall
from copy import deepcopy
from functools import partial
from scipy.special import comb
from itertools import combinations

jl = juliacall.newmodule("pProject")

class FISTAOED(OEDAlgorithm):
    def __init__(self, max_p_steps = 20, pre_tol = 1e-15, update_freq = 50, solver = "SLSQP", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.opts = {"maxiter": int(self.max_iters)}
        self.max_iters_digits = int(np.log10(self.max_iters)+1)
        
        self.max_p_steps = max_p_steps
        self.max_p_steps_digits = int(np.log10(self.max_p_steps)+1)
        self.pre_tol = pre_tol
        self.update_freq = update_freq
        
        self.design = None
        self.global_design = None
        
        self.solver = solver
        self.vals = {}
        self.ws = []
        
    def solve(self, target, z0, p = 1, p_old = 1):

        prox = self.prox(target = target, p = p)
        eva = self.J.eval
        jac = self.J.jac
        hess = self.J.hess
        
        if p == 1:

            # Warm start via SLSQP, which is fairly fast and robust for convex (i.e. p = 1) problems
            
            
            
            update_freq = int(self.update_freq)
            
            bnd = Bounds(lb = 0, ub = 1, keep_feasible = True)
            cns = LinearConstraint(A = np.ones(self.m_sensors), ub = target)
            res = minimize(fun = eva, jac = jac, \
                   x0 = z0, bounds = bnd, constraints = cns, \
                   tol = self.tol, options = self.opts)
            
            z = res["x"]
            #print(z0)
            #print(res)
            
        else:

            ip = 1/p
            
            eva = lambda z: self.J.eval(z**ip)
            jac = lambda z: ip * self.J.jac(z**ip) * z**(ip-1)

            update_freq = int(self.update_freq ** p)

            z = z0**(p / p_old)
            z0 = z
    
            flag_break = False
            change = np.inf
            val = np.inf
            val0 = np.inf
            val_diff = np.inf
            penalty = 1
            self.vals[p] = []
    
            def nrm(A):
                return np.linalg.norm(A)
                #return np.linalg.norm(np.linalg.norm(A,axis=0))
            
            for i in range(int(self.max_iters)):
    
                # Compute jacobian for step size and p-Hessian
                if p == 1:
                    zip0 = z
                else:
                    zip0 = z**ip
                j = jac(zip0)
    
                # At fixed intervals, recompute Lipschitz estimate and test for stopping
                if not i % update_freq or i < max(update_freq // 2,10):
                    change = np.linalg.norm(z-z0)
                    
                    if i > 100 and change < self.tol:# or (val_diff > 0 and val_diff < self.tol)):
                        print("\nNo change, breaking...")
                        flag_break = True
                        break
                    
                    if p == 1:
                        L = nrm(hess(z))
                    else:
                        zip1 = z**(ip-1)
                        with np.errstate(invalid='ignore',divide='ignore'):
                            zip2 = z**(ip-2)
                            zip2[np.isnan(zip2) | np.isinf(zip2)] = 0
                        
                        L = nrm( \
                                ip **2 * zip1.reshape(-1,1) * hess(zip0) * zip1 \
                              - ip * (ip - 1) * np.diag(j * zip2) \
                            )
    
                    L *= 1 + 1/(i+1)

                if np.isinf(penalty*L):
                    print("\nInfinite penalty, breaking...")
                    break
                    
                print("\r","ISTA for p = ",p,": ",str(i).zfill(self.max_iters_digits),"/",self.max_iters, \
                      ", L = ",penalty*L,", val = ",val,", valdiff = ",val_diff,", change = ",change,sep="",end="")
    
                z00 = z0
                z0 = z
                z = z - 1 / (penalty * L) * j
                if np.any(np.isnan(z) | np.isinf(z)):
                    print("\nInvalid gradient step, breaking...")
                    break
                z = prox(x = z)
                if np.any(np.isnan(z) | np.isinf(z)):
                    print("\nInvalid prox step, breaking...")
                    break
                    
                val = eva(z)
                val_diff = val0 - val
    
                if val_diff < 0:
                    penalty *= 1.2
                    z = z0
                    z0 = z00
                else:
                    penalty = max(1,penalty/1.01)
                    self.ws.append(self.insert(z))
                    val0 = deepcopy(val)
                    self.vals[p].append(val)
    
            if not flag_break:
                print("\nISTA did not finish after ",i," iterations for p = ",p,", change: ",change,"...",sep="")
            
        return z
    
    def prox(self, target):
             
        def prox_unit_cube(x,alpha):
            return np.fmax(np.fmin(x,1),0)
        
        def prox_psum_box(x,n,p):
            if p == 1:
                return prox_simplex_box(x,n)
            x = jl.pProject(y,p,n**p)
            return

        def prox_simplex_box(x,n): # Projects onto [0,1]^d intersected with the (filled) simplex scaled by a factor n (i.e. 0<=x<=1 and sum(x)<=n).
            assert n<=len(x), "n must be no greater than the length of the input x \
                (currently n = " + str(n) + ", len(x) = " + str(len(x)) + ")..."
            y = prox_unit_cube(x,None)
            if np.sum(y)<=n:
                return y
            else:
                f = lambda lam: np.sum(prox_unit_cube(x - lam,None)) - n
                try:
                    lam = bisect(f,0,np.amax(x))
                except:
                    print(f(0))
                    print(f(np.amax(x)))
                    print(np.amax(x))
                    print(x)
                    assert 0
                return prox_unit_cube(x - lam,None)
            
        prx = partial(prox_simplex_box, n = target)
        #if p == 1:
        return lambda x: prx(x)
        #return lambda tau, x: prx(x**p)**(1/p)
        
    def Opt(self, target, w0 = None, P = 3, \
                  enforce_bounds = True, continuation = True, only_global = False, verbose = True):
        
        target = int(target)

        self.target_number_of_sensors = target
        self.verbose = verbose
        self.dom_indices = np.array([],dtype=int)
        self.red_indices = np.array([],dtype=int)
        
        if w0 is None:
            w0 = np.ones(self.m_sensors) * target / self.m_sensors
            
        w = self.solve(target = target, z0 = w0)

        w = self.set_binary(w, target = target, set_red_dom = enforce_bounds)
        self.global_design = deepcopy(w)
        
        if only_global:
            self.design = deepcopy(w)
            print("Only outputting global optimum and terminating...")
            return
        
        self.design_sequence = []
        self.ps = []

        p = 1
        p_fac = (P-1)/P
        
        def sround(string):
            return "{:.2e}".format(string)

        z = w[self.free_indices]
        flag_success = False
        for i in range(self.max_p_steps):

            p_old = p
            p = p * p_fac
            ip = 1/p

            if self.verbose:
                print("\r","Running for target ",target,": ",str(i).zfill(self.max_p_steps_digits), "/", int(self.max_p_steps), \
                      ", p = ",sround(p), \
                      ", reds = ",int(len(self.red_indices)), " doms = ",int(len(self.dom_indices)), \
                      sep="",end="\n")
            
            z = self.solve(target = target, z0 = z, p = p, p_old = p_old)

            if enforce_bounds:
                self.red_indices = np.union1d(self.red_indices,self.insert(z)==0)
                self.update()
    
                z = z[self.free_indices]
            
            w = self.insert(z) #**ip
            self.design_sequence.append(w)
            self.ps.append(p)
            self.design = w

            if self.nonb(w) < self.tol:
                self.vprint("\nFound binary design for target " + str(target) + ", terminating...")
                flag_success = True
                break
        
        if not flag_success:
            self.vprint("\nNo binary design found for target " + str(target) + "...")
                
        if enforce_bounds:
            self.dom_indices = np.array([],dtype=int)
            self.red_indices = np.array([],dtype=int)
            
            
