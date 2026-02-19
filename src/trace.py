import numpy as np
from scipy.linalg import cholesky as chol
from scipy.linalg import solve
from ngsolve import TaskManager

class trace:
    def __init__(self, Q, R, W, m_sensors, m_obs, basis_change_matrix, Roffset = None, offset = 0, rescale = False):
        
        self.stored_w = None
        
        self.ell = R.shape[0]
        self.m = R.shape[1]
        self.m_sensors = m_sensors
        self.m_obs = m_obs

        self.R = R
        if Roffset is None:
            self.Roffset = np.eye(self.ell)
        else:
            self.Roffset = Roffset
            
        self.W = W

        self.basis_change_matrix = basis_change_matrix
        self.basis_change_matrix_half = chol(self.basis_change_matrix, lower = True)
    
        if rescale:
            self.rescaled = False
        else:
            self.rescaled = True
            self.offset = offset
            self.mult = 1
        
        self.L = None
        self.LIB = None
        self.val0 = self.eval(np.zeros(self.m_sensors))
        
    def update(self, w):
        
        # Recomputes the reduced-rank matrix L as necessary.
        # Also recomputes the inverse (L+I)^{-1}
        # Compute the matrix (L+I)^{-1} @ basis_change_matrix_half
        
        if self.stored_w is None or np.any(w != self.stored_w):
            try:
                self.stored_w = w
                self.L = (self.R*self.W(w))@self.R.T.conj()
                self.L += self.Roffset
                self.LIB = solve(self.L, self.basis_change_matrix_half, check_finite=True, assume_a='pos')
            except:
                self.L = None
                self.LIB = None
        return
            
    def rescale(self):
        self.rescaled = True
        self.offset = self.eval(np.ones(self.m_sensors))
        self.mult = 1 / self.eval(np.zeros(self.m_sensors))
        return
    
    def eval(self, w):
        
        # Update L if needed.
        self.update(w)
        
        if self.L is None:
            return np.inf
        
        # Rescale on first run.
        if not self.rescaled:
            self.rescale()
        
        return self.mult * (np.trace((self.LIB@self.basis_change_matrix_half.T.conj())) + self.offset)
     
    def jac(self, w):
        
        # Update L if needed.
        self.update(w)
        
        if self.L is None:
            return np.inf * np.ones(self.m_sensors)
        
        # Rescale on first run.
        if not self.rescaled:
            self.rescale()

        # The Jacobian is the (sum of) 2-norms of each (m_sensort-th) column of LIB.T@R
        norms = np.real(np.linalg.norm(self.LIB.T.conj()@self.R, axis=0))**2
        return -self.mult * np.array([np.sum(norms[k::self.m_sensors]) for k in range(self.m_sensors)])
    
    def hess(self, w):
        # Update L if needed.
        self.update(w)
        
        # Rescale on first run.
        if not self.rescaled:
            self.rescale()
            
        R = self.R
        LIR = solve(self.L, R, check_finite=True)
        
        LIBB = self.LIB@self.basis_change_matrix_half.T.conj()
        LIBB = LIBB.T.conj()
        
        slicer = lambda A, i: np.roll(A,-self.m_sensors * i)[:,:self.m_sensors]
        
        H = np.zeros((self.m_sensors,self.m_sensors))
        with TaskManager():
            for K in range(self.m_obs):
                LIRK = slicer(LIR, K)
                RK = slicer(R, K)
                for L in range(K,self.m_obs):
                    LIRL = slicer(LIR, L)
                    
                    H += (LIRK.T.conj() @ self.basis_change_matrix @ LIRL) * \
                         (RK.T.conj() @ LIRL)
                
        H = H + H.T.conj() - np.diag(np.diag(H))
        return 2 * H
        
    def hess_matvec(self, w, z):
        
        # Update L if needed.
        self.update(w)
        
        # Rescale on first run.
        if not self.rescaled:
            self.rescale()

        # The Hessian is a Hadamard product, which can be efficiently computed as the diagonal of a matrix 
        # (A o B)z = diag(AD_z B^T)
        
        R = self.R
        LIR = solve(self.L, R, check_finite=True)
        
        LIBB = self.LIB@self.basis_change_matrix_half.T.conj()
        LIBB = LIBB.T.conj()
        
        slicer = lambda A, i: np.roll(A,-self.m_sensors * i)[:,:self.m_sensors]
        
        zout = np.zeros_like(z)
        with TaskManager():
            for K in range(self.m_obs):
                LIRK = slicer(LIR, K)
                RK = slicer(R, K)
                for L in range(K,self.m_obs):
                    LIRL = slicer(LIR, L)
                    RL = slicer(R, L)
                    H = (RL * z) @ RK.T.conj()
                    H = LIBB @ H
                    zout += (1 + L > K) * np.array([np.inner(H@LIRL[:,k],LIRK[:,k]) for k in range(self.m_sensors)])
                
        return 2 * zout