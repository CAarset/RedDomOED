import numpy as np
from scipy.linalg import cholesky as chol
from scipy.linalg import solve
from ngsolve import TaskManager

class trace:
    def __init__(self, Q, R, W, m_sensors, m_obs, basis_change_matrix, Roffset = None, offset = 0):
        
        self.stored_w = None
        
        self.ell = R.shape[0]
        self.m = R.shape[1]
        self.m_sensors = m_sensors
        self.m_obs = m_obs

        self.R = R
        self.Roffset = Roffset
            
        self.W = W
        self.offset = offset

        self.basis_change_matrix = basis_change_matrix
        self.basis_change_matrix_half = chol(self.basis_change_matrix, lower = True)
        self.basis_change_matrix_halfT = self.basis_change_matrix_half.T # Known issue: Notation is backwards from that used in the paper (halfT is half and vice versa)
        
        #self.L = None
        #self.LIB = None

    #@property
    #def w(self):
    #    return self._w

    #@w.setter
    #def w(self, value):
    #    self._w = value
    #    self._clearLIB()

    #def _clearLIB(self):
    #    self._LIB = None

    #@property
    def LI(self,w,A):
        #if self._LIB is None:
        L = (self.R*self.W(w))@self.R.T
        if self.Roffset is not None:
            L += self.Roffset
        else:
            # Faster than L = L + np.eye(ell)
            L[np.diag_indices(self.ell)] += 1
        return solve(L, A, check_finite=False, assume_a='pos')
        #self._LIB =#return self._LIB

    def LIBB(self,w):
        return self.LI(w = w, A = self.basis_change_matrix)

    def LIR(self,w):
        return self.LI(w = w, A = self.R)
    
    def eval(self, w):
        return self.LIBB(w).trace() + self.offset
     
    def jac(self, w):
        # The Jacobian is the (sum of) 2-norms of each (m_sensort-th) column of LIB.T@R
        norms = -((self.basis_change_matrix_halfT@self.LIR(w))**2).sum(0)
        return norms.reshape(self.m_obs,self.m_sensors).sum(0)
    
    def hess(self, w):
        # The Hessian can be expressed as a (sum of) Schur (aka Hadamard) product
        # on the form (L_w^{-1}
        
        R = self.R
        LIR = self.LIR(w)

        Rk = lambda k: R[:,(self.m_sensors * k):(self.m_sensors * (k + 1))]
        LIRk = lambda k: LIR[:,(self.m_sensors * k):(self.m_sensors * (k + 1))]
        
        H = np.zeros((self.m_sensors,self.m_sensors))
        for k in range(self.m_obs):
            LIRkTC = LIRk(k).T@self.basis_change_matrix
            for l in range(k,self.m_obs):
                HH = (LIRkTC@LIRk(l)) * (Rk(k).T@LIRk(l))
                H += HH
                if l > k:
                    H += HH.T
        return 2 * H
        
    def hess_matvec(self, w, v):
        # Efficiently applies the Hessian matrix by employing the Schur product identities
        # (A1 * A2)x = diag( (x * A1) @ A2.T) = sum_columns( (x * A1) * A2)
        
        R = self.R
        LIR = self.LIR(w)

        Hl = (self.W(v) * LIR) @ LIR.T
        Hl = Hl @ self.basis_change_matrix
        Hl = np.sum((Hl@LIR)*R,axis=0)
        vout = np.sum(Hl.reshape(self.m_obs,self.m_sensors),axis=0)
        
        return 2 * vout
