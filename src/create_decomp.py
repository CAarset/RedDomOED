import numpy as np
from scipy.sparse import eye
from time import time
from ngsolve import TaskManager
from functools import partial

from randomized_subspace_iteration import *
#from rsvd import rsvd
#from lrQR import lrQR

class create_decomp:
    def __init__(self, F, FT, m, n, \
                 tol = 1e-15, target_rank = 100, mode = "automatic", \
                 number_of_low_rank_iterations = 2, oversampling_parameter = 2, \
                 complex_source = False, complex_data = True):
        
        self.F = F
        self.FT = FT
        
        self.m = m
        self.n = n
                              
        self.tol = tol
        self.target_rank = target_rank
        self.number_of_low_rank_iterations = number_of_low_rank_iterations
        self.oversampling_parameter = oversampling_parameter
        
        self.mode = mode
        
        self.actual_spectrum = None
        self.full_spectrum = None
        
        self.SVD = None
        self.full_spectrum = None
                              
        self.Q = None
        self.R = None
        self.ell = None
        
    def SVD_to_QR(self):
        
        SVD = self.SVD

        self.Q = SVD["U"]
        self.R = (SVD["Vh"].T*SVD["S"]).T
                
    def decomp(self):
        
        decompstart = time()
        
        if self.m >= self.n:
                
            RSI = rsi(op = self.F, opT = self.FT, \
                      m = self.m, n = self.n, \
                      target_rank = self.target_rank, \
                      oversampling_parameter = self.oversampling_parameter, \
                      number_of_low_rank_iterations = self.number_of_low_rank_iterations)

            V, S, Uh = RSI.randomized_subspace_iteration()

            U = Uh.T.conj()
            Vh = V.T.conj()

        else:

            RSI = rsi(op = self.FT, opT = self.F, \
                      m = self.n, n = self.m, \
                      target_rank = self.target_rank, \
                      oversampling_parameter = self.oversampling_parameter, \
                      number_of_low_rank_iterations = self.number_of_low_rank_iterations)

            U, S, Vh = RSI.randomized_subspace_iteration()

        self.SVD = {"U": U, "S": S, "Vh": Vh}
        self.full_spectrum = RSI.full_spectrum
        
        self.decomptime = time() - decompstart
        
        self.ell = len(S)
        self.SVD_to_QR()