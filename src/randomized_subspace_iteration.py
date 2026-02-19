import numpy as np
from scipy.linalg import qr, svd, eig, eigh, khatri_rao
from scipy.sparse import csr_matrix
from copy import deepcopy

class rsi:
    def __init__(self, op, opT, m, n, target_rank, oversampling_parameter=0, tol=1e-15, \
                 number_of_low_rank_iterations=2):#, complex_source = False, complex_data = False):

        self.op = op
        self.opT = opT
        self.m = m
        self.n = n
        
        self.tol = tol
        
        self.number_of_low_rank_iterations = number_of_low_rank_iterations
        self.target_rank = target_rank
        l = target_rank + oversampling_parameter
        
        #if complex_source:
        #    self.dtype = np.complex64
        #    self.O = np.random.normal(size=(self.n, l)) + 1j * np.random.normal(size=(self.n, l)) #/ np.sqrt(2)
        #    self.in_map = lambda f: f
        #else:
        self.dtype = np.float32
        self.O = np.random.normal(size=(self.n, l))
        
        #self.in_map = lambda f: f.real
        #if complex_data:
        #self.out_map = lambda g: g
        #else:
        #    self.out_map = lambda g: g.real

        # Quick-reference low-rank tools.
        self.QR = lambda A: qr(A, mode = 'economic')
        self.SVD = lambda A: svd(A, full_matrices = False)
        
        def SVDsym(A):
            A = (A + A.T.conj())/2
            S, U = eigh(A)
            return U, S
        
        self.SVDsym = SVDsym

    def randomized_subspace_iteration(self):
        op = self.op
        opT = self.opT

        qr = self.QR
        svd = self.SVD

        O = deepcopy(self.O)
        O = op(O)
        Q, _ = qr(O)
        for i in range(self.number_of_low_rank_iterations):
            print("\r","Randomized subspace iteration ",i,"/",self.number_of_low_rank_iterations,"...",sep="",end="")
            O = opT(Q)
            Q, _ = qr(O)
            O = op(Q)
            Q, _ = qr(O)
        B = opT(Q)
        U, S, Vh = svd(B.T.conj())
        
        self.full_spectrum = deepcopy(S)
        #if np.max(S[self.target_rank:]) > 1e-15 * S[0]:
        #    print("Warning: Cutting off significant eigenvalues in rSVD!", np.max(S[self.target_rank:]), "vs.", S[0])
        target_rank = np.sum(S > 1e-10 * S[0])#self.tol * S[0])
        if target_rank < self.target_rank:
            print("Lowered target from", self.target_rank, "to", target_rank)
        else:
            print("Full RSVD rank, consider higher target (", S[-1], "vs.", S[0],")")
        self.target_rank = min(self.target_rank,target_rank)
            
        U, S, Vh = U[:,:self.target_rank], S[:self.target_rank], Vh[:self.target_rank,:]
        U = Q @ U
        
        print("-"*40)
        print((U.shape,S.shape,Vh.shape))
        print("-"*40)
        print(U)
        print("-"*40)
        print(S)
        print("-"*40)
        print(Vh)
        print("-"*40)
        return U, S, Vh
