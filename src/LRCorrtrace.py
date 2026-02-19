import numpy as np

import autograd.numpy as np  # Thinly-wrapped numpy
import autograd.scipy as scipy

from autograd.scipy.linalg import cholesky as chol
from autograd.numpy.linalg import solve
from autograd import grad, jacobian, hessian

from ngsolve import TaskManager

from matrix_operators import *

class trace:
    def __init__(self, R, Rcorr, W, m_sensors, m_obs, basis_change_matrix, Roffset = None, offset = 0):
        
        self.stored_w = None
        
        self.ell = R.shape[0]
        self.ellsquare = self.ell**2
        self.m = R.shape[1]
        self.m_sensors = m_sensors
        self.m_obs = m_obs

        self.R = R
        self.Roffset = Roffset
            
        self.W = W
        self.offset = offset

        self.ellcorr = Rcorr.shape[0]
        self.elllsquare = R.shape[1]
        
        self.Rcorr = Rcorr
        self.Roffset = Roffset

        self.basis_change_matrix = basis_change_matrix
        self.basis_change_matrix_half = chol(self.basis_change_matrix, lower = True)
        self.basis_change_matrix_halfT = self.basis_change_matrix_half.T # Known issue: Notation is backwards from that used in the paper (halfT is half and vice versa)
        
        self.jacobian_eval = jacobian(self.eval)

    #@property
    #def w(self):
    #    return self._w

    #@w.setter
    #def w(self, value):
    #    self._w = value
    #    self._clearLIB()

    #def _clearLIB(self):
    #    self._LIB = None

    def RcorrTslice(self, i):
        return self.Rcorr[:,(self.ell*i):(self.ell*(i+1))].T
        
    def L(self, w):
        
        if w is None:
            return self.Roffset
            
        L = (self.R*self.W(w))@self.R.T
        if self.Roffset is not None:
            L += self.Roffset
        return L

    #@property
    def LI(self,w,A):

        if w is None:
            return A
        
        L = self.L(w)
        L = kroenecker(L,L)
        L = L(self.Rcorr.T)
        L = self.Rcorr@L

        # Faster than L = L + np.eye(ellcorr)... but not viable for autograd.
        #LL[np.diag_indices(self.ellcorr)] += 1
        L = L + np.eye(self.ellcorr)
        
        return solve(L, A)#, check_finite=False, assume_a='pos')
        
    def LIB(self,w):
        return self.LI(w = w, A = self.basis_change_matrix_half)
    
    def LIBB(self,w):
        return self.LI(w = w, A = self.basis_change_matrix)
        
    def LIR(self,w):
        return self.LI(w = w, A = self.Rcorr)
    
    def eval(self, w):
        #if (w < 0).any():
        #    return np.inf
        return self.LIBB(w).trace() + self.offset
     
    def jac(self, w):
        # Use autograd for derivative
        return self.jacobian_eval(w)

        ## The Jacobian is the (sum of) 2-norms of ???
        #L = self.L(w)
        #LIB = self.LIB(w)
        #
        #Bhat = np.zeros((self.ell,self.ell))
        #Btil = np.empty((self.ell,self.ell))
        #for i in range(self.ell):
        #    RLIBi = self.RcorrTslice(i)@LIB
        #    for j in range(i,self.ell):
        #        if i == j:
        #            RLIBj = RLIBi
        #        else:
        #            RLIBj = self.RcorrTslice(j)@LIB
        #        RLR = RLIBi @ RLIBj.T
        #        Btil[i,j] = (L@RLR).trace()
        #        Bhat += L[i,j] * RLR
        #        if j > i:
        #            Bhat += L[j,i] * RLR.T
        #
        #Btil = Btil + Btil.T - np.diag(Btil.diagonal())
        #
        ##Bh = chol(Bhat+Btil, lower = True)
        ##norms = - ((Bh@self.R)**2).sum(0)
        #
        ##Bhathalf = chol(Bhat, lower = True)
        ##Btilhalf = chol(Btil, lower = True)
        #
        ##norms = -((Bhathalf@self.R)**2).sum(0) - ((Btilhalf@self.R)**2).sum(0)
        #
        #norms = - diagg(self.R.T, (Bhat+Btil)@self.R)
        #return norms.reshape(self.m_obs,self.m_sensors).sum(0)
    
    def hess(self, w):
        raise Exception("Not implemented")
        
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