import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
from scipy.linalg import cholesky as chol

from lrQR import *
from LRtrace import trace

import dill as pickle
from copy import deepcopy
from functools import partial

from ngsolve import TaskManager, GridFunction
from util import power_iteration_matrix
from create_decomp import *

from os import listdir, makedirs
from os.path import isfile, join
from time import time
from datetime import timedelta

class OEDAlgorithm:
    def __init__(self, mmaker,
                    omegas = [50],
                    noise_level = 1e-2, tol = 1e-15, max_iters = int(1e3),
                    jtol = 1e-2,
                    shape = "round", target_rank = 100, mode = "automatic",
                    verbose = True, samples = int(1e3), noise_type = "average", **kwargs):
        
        self.rng = np.random.default_rng(199)
        
        self.w = None
        self.w_old = None
        
        # Grab mesh- and grid makers
        self.mmaker = mmaker

        # PDE arguments (should change to a kwargs format)
        self.omegas = np.sort(np.unique(omegas))
        
        # Grab important functions
        self.C = mmaker.C
        self.A = mmaker.A
        self.AT = mmaker.AT
        self.F = partial(mmaker.F, omegas = self.omegas)
        self.FT = partial(mmaker.FT, omegas = self.omegas)
        self.FC = partial(mmaker.FC, omegas = self.omegas)
        self.CFT = partial(mmaker.CFT, omegas = self.omegas)

        self.real_to_ngs = mmaker.real_to_ngs
        self.ngs_to_real = mmaker.ngs_to_real
        self.coeff_to_ngs = mmaker.coeff_to_ngs
        self.ngs_to_coeff = mmaker.ngs_to_coeff

        # mesh and FESes easily available
        self.mesh = self.mmaker.mesh
        self.fes = self.mmaker.fes
        self.cofes = self.mmaker.cofes
        
        # Work out actual problem dimensions

        self.domain_is_complex = self.mmaker.domain_is_complex
        self.data_is_complex = self.mmaker.data_is_complex
        
        self.n = mmaker.n

        self.m_obs = len(omegas)
        if self.data_is_complex:
            self.m_obs *= 2
        self.m_sensors = self.mmaker.m_sensors
        self.m_sensors_actual = self.mmaker.m_sensors_actual
        self.m = int(self.m_sensors * self.m_obs)

        self.dtype = self.mmaker.dtype
        
        self.tol = tol # Tolerance for solvers
        self.jtol = jtol # Relative tolerance for numerical ordering of gradient
        self.max_iters = max_iters
        self.target_rank = target_rank(m = self.m, n = self.n)
        self.mode = mode
        self.shape = shape
        self.samples = samples
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        self.current_m = deepcopy(self.m)
        self.current_free = deepcopy(self.m_sensors)
        self.current_target = deepcopy(self.m_sensors)
        
        self.design = np.zeros(self.m_sensors)

        self.free_indices = np.arange(self.m_sensors, dtype = int)
        self.dom_indices = np.array([], dtype = int)
        self.red_indices = np.array([], dtype = int)

        self.fixed_doms = 0
        self.fixed_reds = 0

        self.out_flag = [0,0,"Failed"]
        self.verbose = verbose
        
        self.actual_spectrum = None
        self.full_spectrum = None
        
        def vprint(str):
            if self.verbose:
                print(str)
        self.vprint = vprint

        outputs_directory = "opt_outputs/"
        makedirs(outputs_directory, exist_ok = True)
        self.outputs_directory = outputs_directory
        
        decomps_dumps_directory = "decomps/"
        makedirs(decomps_dumps_directory, exist_ok = True)
        self.decomps_dumps_directory = decomps_dumps_directory

        target_filename = "msensor_" + str(self.m_sensors) + "_mobs_" + str(self.m_obs) + \
                          "_omegas_" + str(self.omegas) + \
                          "_noisetype_" + str(self.noise_type) + "_noiselevel_" + str(round(-np.log10(self.noise_level),2)) + \
                          "_" + self.mmaker.target_filename
        #target_filename = "msensor_" + str(self.m_sensors) + "_mobs_" + str(self.m_obs) + "_shape_" + self.shape + \
        #                  "_omegas_" + str(self.omegas) + "_target_" + str(self.target_rank) + \
        #                  "_noise_" + self.noise_type + "_" + \
        #                  self.mmaker.target_filename
        self.target_filename = target_filename
        
        load_flag = False
        # Attempt to find a stored decomposition with the same target
        for filename in listdir(decomps_dumps_directory):
            if isfile(join(decomps_dumps_directory, filename)):
                if target_filename == filename:
                    with open(join(self.decomps_dumps_directory, filename), "rb") as input_file:
                        obj = pickle.load(input_file)
                        self.vprint("Successfully loaded stored decomposition!")
                        self.ell = obj["ell"]
                        self.R = obj["R"]
                        self.Q = obj["Q"]
                        
                        self.full_spectrum = obj["full_spectrum"]
                        
                        self.norm = obj["norm"]
                        
                        self.basis_change_matrix = obj["basis_change_matrix"]
                        self.CQmat = obj["CQmat"]
                        
                        self.offset = obj["offset"]
                        self.decomptime = obj["decomptime"]
                        self.data_variance = obj["data_variance"]
                        
                        load_flag = True
                        break
        if not load_flag:
            filename = decomps_dumps_directory + target_filename
            self.vprint("Data file " + filename + " does not exist, creating decomposition " + \
                         "with target " + str(self.target_rank) + "...")
            
            # Estimation of data size for relative noise level purposes.
            data_variance = np.zeros(self.m)
            
            if self.noise_type.casefold() == "continuous".casefold():
                print("Computing continuous noise level...")
                C = self.C
                A = self.A
                AT = self.AT
                
                def forw_back(f):
                    p = C(f)
                    p = C(p)
                    q = GridFunction(self.fes)
                    for omega in self.omegas:
                        q.vec.FV().NumPy()[:] += AT(u = A(f = p, omega = omega), omega = omega).vec.FV().NumPy()[:]
                    return q
                
                self.data_variance = self.mmaker.trace(Cov = forw_back)
                self.data_variance /= self.n                
                print("Continuous noise level:",self.data_variance)
                self.data_variance *= self.mmaker.sensor_norm ** 2
                print("Norm-scaled continuous noise level:",self.data_variance)
            else:
                
                samples = self.samples
                for samp in range(int(samples)):
                    print("\rDrawing sample ",samp,"/",samples,"...",sep="",end="")
                    s = self.mmaker.sample(self.C)
                    gs = self.F(s)
                    data_variance += np.abs(gs)**2 / samples
                    
                if self.noise_type.casefold() == "exp".casefold():
                    max_data_variance = np.max(data_variance)
                    min_data_variance = np.min(data_variance)

                    x_norms = np.linalg.norm(self.grid, axis = 1)

                    a = (np.log(max_data_variance)-np.log(min_data_variance)) / (np.max(x_norms) - np.min(x_norms))
                    b = max_data_variance * np.exp(a * np.min(x_norms))

                    self.data_variance = b * np.exp(-a * x_norms)
                elif self.noise_type.casefold() == "quad".casefold():
                    max_variant_sensor = np.argmax(data_variance)
                    max_data_variance = data_variance[max_variant_sensor]

                    x_norms = np.linalg.norm(self.grid, axis = 1)

                    a = max_data_variance * x_norms[max_variant_sensor]**2
                    
                    self.data_variance = a / x_norms**2
                elif self.noise_type.casefold() == "average".casefold():
                    self.data_variance = np.mean(data_variance)
                else:
                    self.data_variance = data_variance
                
            # Create QR decomposition
            decomp_maker = create_decomp(F = self.Forward, FT = self.Adjoint, \
                                         m = self.m, n = self.n, \
                                         tol = self.tol, target_rank = self.target_rank, \
                                         mode = self.mode)
            decomp_maker.decomp()
            
            self.Q = decomp_maker.Q
            self.R = decomp_maker.R
            self.ell = decomp_maker.ell
            self.actual_spectrum = decomp_maker.actual_spectrum
            self.full_spectrum = decomp_maker.full_spectrum
            self.decomptime = decomp_maker.decomptime
            
            Q, R = self.Q, self.R
            
            self.vprint("Dimension reduced from " + str((self.m, self.n)) + \
                        " to " + str(self.ell) + " in " + str(timedelta(seconds = self.decomptime)))

            # Norm estimation to make use of relative noise level
            norm_square = power_iteration_matrix(lambda x: Q@(R@(R.T.conj()@(Q.T.conj()@x))), n = self.n, iters = 100)
            self.norm = np.sqrt(norm_square)
            print("Forward operator norm estimate:",self.norm, "vs.", self.data_variance)
            
            # The basis change matrix QC_{prior}^2Q^T
            with TaskManager():
                Cmat = np.empty((self.n,self.ell))
                CQmat = np.empty((self.n,self.ell))
                for i in range(self.ell):
                    CQ = self.mmaker.real_to_ngs(self.Q[:,i])
                    CQ = self.C(CQ)
                    CQmat[:,i] = self.mmaker.ngs_to_real(CQ)
                    CQ = self.C(CQ)
                    Cmat[:,i] = self.mmaker.ngs_to_real(CQ)
                Cmat = Q.T.conj() @ Cmat
            self.CQmat = CQmat
            self.basis_change_matrix = Cmat
            
            self.offset = self.mmaker.tracePrior - np.trace(self.basis_change_matrix)
            
            if self.target_filename is not None:
                obj = {"ell": self.ell, \
                       "R": self.R, "Q": self.Q, \
                       "norm": self.norm, "full_spectrum": self.full_spectrum,
                       "basis_change_matrix": self.basis_change_matrix, "CQmat": self.CQmat,
                       "offset": self.offset, "decomptime": self.decomptime,
                       "data_variance": self.data_variance}
                with open(filename, "wb") as output_file:
                    pickle.dump(obj, output_file)    
                self.vprint("Successfully stored decomposition.")

        # Scale R based on desired relative noise level (does not require re-run).
        self.R_unscaled = self.NoiseCovHalf(self.R.T, include_noise_level = False).T
        self.R = 1 / np.sqrt(self.noise_level) * self.R

        self.Rfree = deepcopy(self.R)
        self.I_plus_RdomR = np.eye(self.ell)
        
        J = trace(Q = self.Q, R = self.R, W = self.W, m_sensors = self.m_sensors, m_obs = self.m_obs, \
                  basis_change_matrix = self.basis_change_matrix, Roffset = None, offset = self.offset)
        self.J_init = J
        self.J = J
        
        self.Bh = self.J_init.basis_change_matrix_half
        self.Bh0 = self.J_init.basis_change_matrix_half
    
    def W(self, w):
        return np.tile(w,self.m_obs)
    
    def NoiseCovHalf(self, g, include_noise_level = False):
        g_noise = g * np.sqrt(self.data_variance)
        if include_noise_level:
            g_noise *= np.sqrt(self.noise_level)
        return g_noise
        
    def NoiseCovInverseHalf(self, g, include_noise_level = False):
        g_noise = g / np.sqrt(self.data_variance)
        if include_noise_level:
            g_noise /= np.sqrt(self.noise_level)
        return g_noise
    
    def Forward(self, X):
        Y = np.empty((self.m, X.shape[1]))
        with TaskManager():
            for i in range(X.shape[1]):
                x = X[:,i]
                Y[:,i] = self.NoiseCovInverseHalf(self.FC(x))
        return Y

    def Adjoint(self, Y):
        X = np.empty((self.n, Y.shape[1]))
        with TaskManager():
            for i in range(Y.shape[1]):
                g = Y[:,i]
                X[:,i] = self.CFT(self.NoiseCovInverseHalf(g))
        return X
    
    def design_to_cov(self, w, fac = 1, half = False, low_rank = True):
        
        Q = self.Q
        R = self.R
        mmaker = self.mmaker
        
        LIB = (R*self.W(w)) @ R.T.conj() + fac * np.eye(self.ell)
        LIB = solve(LIB, self.Q.T.conj(), check_finite=True, assume_a='pos')
        
        def cov(r):
            y = mmaker.Kinv(r)
            y = mmaker.Mchol["apply_h"](y)
            y = y + Q@((LIB-Q.T.conj())@y)
            y = mmaker.Mchol["apply_hT"](y)
            y = mmaker.Kinv(y)
            return y
        
        if half:
            cov_decomp = create_decomp(F = cov, FT = cov, m = self.n, n = self.n, \
                         tol = self.tol, target_rank = self.target_rank, \
                         mode = self.mode)
            cov_decomp.do()
            
            return lambda x: cov_decomp.SVD["U"]@(np.sqrt(cov_decomp.SVD["S"]) * (cov_decomp.SVD["Vh"]@x))
        
        elif not low_rank:
            mmaker = self.mmaker
            
            def cpst(r):
                f = mmaker.real_to_ngs(r)

                # Prior inverse
                Cf = mmaker.Mchol["solve"](mmaker.K@mmaker.ngs_to_coeff(f))
                Cf = mmaker.Mchol["solve"](mmaker.K@Cf)
                Cf = mmaker.coeff_to_ngs(Cf)
                
                # Forward-adjoint
                g = self.F(f)
                g = self.NoiseCovInverseHalf(g)
                g = self.W(w) * g
                g = self.NoiseCovInverseHalf(g)
                f = self.FT(g)
                
                # Combine
                f.vec.data += Cf.vec.data
                
                return mmaker.ngs_to_real(f)

            Cpst = scipy.sparse.linalg.LinearOperator(shape = (self.n,self.n), matvec = cpst, rmatvec = cpst)
            Preconditioner = scipy.sparse.linalg.LinearOperator(shape = (self.n, self.n), matvec = cov, rmatvec = cov)
            
            return lambda r: scipy.sparse.linalg.cg(Cpst,r, M = Preconditioner)
        else:
            return cov
        
    def design_to_sol(self, w, r = None, f = None, u = None, g = None, add_noise = False, fac = 1):
        assert (r is not None) + (f is not None) + (u is not None) + (g is not None) == 1, "Exactly one of r, f, u and g should be specified!"
        
        Q = self.Q
        R = self.R
        mmaker = self.mmaker
        
        LIR = (R*self.W(w)) @ R.T.conj() + fac * np.eye(self.ell)
        LIR = solve(LIR, R, check_finite=True)
        
        if r is not None:
            f = mmaker.real_to_ngs(r)
        if f is not None:
            g = self.F(f, omegas = self.omegas)
        if u is not None:
            g = self.O(u)
        
        if add_noise:
            raise Exception("Noise addition not currently supported.")
            
        reco = self.NoiseCovInverseHalf(self.W(w) * g)
        reco = self.NoiseCovInverseHalf(reco) * np.sqrt(2) # Why? Scaling issue?
        reco = LIR @ reco
        reco = Q @ reco
        reco = mmaker.real_to_ngs(reco)
        reco = mmaker.C(reco)
        
        return reco        
        
    def Jac(self, w, full = False):
        
        if full:
            LIB = (self.R*self.W(w))@self.R.T.conj() + np.eye(self.ell)
            LIB = solve(LIB, self.Bh0, check_finite=True, assume_a='pos')
            
            norms = np.linalg.norm(LIB.T.conj()@self.R, axis=0)**2
            
            return -np.array([np.sum(norms[i::self.m_sensors]) for i in range(self.m_sensors)])
            
        else:
            LIB = (self.Rfree*self.W(w[self.free_indices]))@self.Rfree.T.conj() + self.I_plus_RdomR
            LIB = solve(LIB, self.Bh, check_finite=True, assume_a='pos')
            
            norms = np.linalg.norm(LIB.T.conj()@self.Rfree, axis=0)**2
            raise Exception("Not properly implemented.")
            return
        
    def update(self, dominant_indices = np.array([], dtype = int), redundant_indices = np.array([], dtype = int)):
        assert not np.intersect1d(dominant_indices, redundant_indices).size, "Indices cannot be both dominant and redundant!"
        self.dom_indices = np.union1d(self.dom_indices, dominant_indices)
        self.red_indices = np.union1d(self.red_indices, redundant_indices)
        self.free_indices = np.setdiff1d(self.free_indices, np.union1d(self.dom_indices,self.red_indices))
        self.current_free = self.free_indices.size
        
        wfree = np.zeros(self.m_sensors)
        wfree[self.free_indices] = 1
        self.Rfree = self.R[:,np.argwhere(self.W(wfree)).ravel()]

        wdom = np.zeros(self.m_sensors)
        wdom[self.dom_indices] = 1
        self.Rdom = self.R[:,np.argwhere(self.W(wdom)).ravel()]

        self.I_plus_RdomR = np.eye(self.ell)
        if self.dom_indices.size:
            self.I_plus_RdomR += self.Rdom@self.Rdom.T

        self.J = trace(Q = self.Q, R = self.Rfree, W = self.W, m_sensors = self.current_free, m_obs = self.m_obs, \
                       basis_change_matrix = self.basis_change_matrix, Roffset = self.I_plus_RdomR, offset = self.offset)
        self.B = self.J.basis_change_matrix
        self.Bh = self.J.basis_change_matrix_half

        #self.current_target = self.target_number_of_sensors - len(self.dom_indices)
        self.current_m = self.m_sensors - len(self.dom_indices) - len(self.red_indices)
        self.fixed_doms += len(dominant_indices)
        self.fixed_reds += len(redundant_indices)
        if 0:#self.current_free:
            C = np.linalg.norm(self.B, ord = 2) * np.linalg.norm(self.Rfree@self.Rfree.T, ord = 2)**2
            self.C0 = np.sqrt(self.current_target) * C
            self.C1 = np.sqrt(self.current_m - self.current_target) * C
            self.C2 = np.sqrt(2 * self.current_target) * C
            self.vprint("C0: " + str(self.C0) + ", C1: " + str(self.C1) + ", C2: " + str(self.C2))
        return
    
    def test_optimality(self, w, J = None, full = False, current_target = None, exit = True):

        if current_target is None:
            if full:
                current_target = int(np.sum(w!=0))
            else:
                current_target = self.current_target
            
        if current_target == 0:
            if np.all(w==0):
                return True
            else:
                return False
            
        if J is None:
            J = self.Jac(w, full = full)
           
        #largest_grad_components = np.argpartition(-J, -current_target)[-current_target:]
        nth_largest_grad_component = np.argpartition(-J, -current_target)[-current_target]
        large_grad_components = np.argwhere(-J > (nth_largest_grad_component + np.sqrt(self.tol)))
        
        if full:
            wfree = w
        else:
            wfree = w[self.free_indices]
        active_sensors = np.nonzero(wfree)
        
        ## If the sets are the same, optimality applies.
        #if not np.setdiff1d(largest_grad_components,active_sensors).size and not np.setdiff1d(active_sensors,largest_grad_components).size:
        
        # Actually, it is enough that the active sensors correspond to any large grad components.
        if not np.setdiff1d(active_sensors,large_grad_components).size:
            self.vprint("Active sensors: " + str(active_sensors))
            self.vprint("Large grad components: " + str(large_grad_components))
            if exit:
                self.vprint("Global minimum found by optimality criterion, exiting!")
                self.out_flag[2] = "Optimality"
                self.update(dominant_indices = np.nonzero(w), redundant_indices = np.nonzero(w==0))
                self.fixed_doms = len(self.dom_indices)
                self.fixed_reds = len(self.red_indices)
            return True
        return False
    
    def nonb(self, w):
        return np.mean(w*(1-w))

    def set_binary(self, w, target, return_jac = False, sort = False, set_red_dom = False):
        # If w is assumed to be the globally optimal solution of the 1-relaxed sensor placement problem,
        # then it will typically have a large number of exactly-0 (and possibly some exactly-1) weights
        # corresponding to small resp. large gradient components.
        # This algorithm identifies such indices, and corrects any that have been set numerically close
        # to, but not exactly equal to, 0 or 1.

        j = self.J_init.jac(w)
        order = np.argsort(j)
        jo = j[order]
        jom0 = jo[target]

        if sort:
            w = w[order]
            j = jo
            
        dom = (w > 1 - self.tol) & ((j - jom0)/np.abs(jom0) <  - self.jtol)
        red = (w < self.tol) & ((j - jom0)/np.abs(jom0) > self.jtol)
        free = np.logical_not(dom) & np.logical_not(red)

        dom = np.argwhere(dom)
        red = np.argwhere(red)
        free = np.argwhere(free)

        w[dom] = 1
        w[red] = 0

        if set_red_dom:
            self.dom_indices = dom
            self.red_indices = red
            self.free_indices = free
            self.vprint("Setting " + str(dom.size) + " dominant and " + str(red.size) + " redundant indices from global solution...")    
            self.update()

        if return_jac:
            return w, j
        else:
            return w

    def insert(self, w):
        wout = np.zeros(self.m_sensors)
        wout[self.free_indices] = w
        wout[self.dom_indices] = 1
        return wout
    
    def Fd(self, f = None, r = None, include_noise = True):
        if include_noise:
            R = self.R
        else:
            R = self.R_unscaled

        assert f is not None or r is not None, "Specify either f or r!"
        if f is not None:
            r = self.ngs_to_real(f)
        g = R.T@(self.Q.T@r)
        return g
    
    def FdT(self, g, output_r = False, include_noise = True):
        if include_noise:
            R = self.R
        else:
            R = self.R_unscaled
        r = self.Q@(R@g)
        if output_r:
            return r
        else:
            return self.real_to_ngs(r)