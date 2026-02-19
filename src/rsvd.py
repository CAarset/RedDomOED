"""=============================================================================
Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

"Finding structure with randomness: Probabilistic algorithms for constructing
approximate matrix decompositions"
============================================================================="""

import numpy as np
# ------------------------------------------------------------------------------

class rsvd:
    """Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    """

    def __init__(self, A, AT, m, n, rank = 100, tol = 1e-15, n_oversamples = None, n_subspace_iters = None, return_range = False):

        self.A = A
        self.AT = AT

        self.m = m
        self.n = n
        self.rank = rank
        self.n_oversamples = n_oversamples
        self.n_subspace_iters = n_subspace_iters
        if n_oversamples is None:
            # This is the default used in the paper.
            self.n_samples = 2 * self.rank
        else:
            self.n_samples = self.rank + n_oversamples
        self.return_range = return_range

    def do(self):
        # Stage A.
        Q = self.find_range()

        # Stage B.
        B = self.AT(Q)
        U, S, Vt = np.linalg.svd(B.T, full_matrices = False)

        # Truncate.
        if self.rank is None:
            mask = np.abs(S) < np.max(np.abs(S)) * min(1, tol)
            U, S, Vt = U[:, mask], S[mask], Vt[mask, :]
        else:
            U, S, Vt = U[:, :self.rank], S[:self.rank], Vt[:self.rank, :]
        U = Q @ U
        
        # This is useful for computing the actual error of our approximation.
        if self.return_range:
            return U, S, Vt, Q
        return U, S, Vt

    # ------------------------------------------------------------------------------

    def find_range(self):
        """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

        Given a matrix A and a number of samples, computes an orthonormal matrix
        that approximates the range of A.

        :param A:                (m x n) matrix.
        :param n_samples:        Number of Gaussian random samples.
        :param n_subspace_iters: Number of subspace iterations.
        :return:                 Orthonormal basis for approximate range of A.
        """

        O = np.random.randn(self.n, self.n_samples)
        Y = self.A(O)

        if self.n_subspace_iters:
            return self.subspace_iter(Y)
        else:
            return self.ortho_basis(Y)

    # ------------------------------------------------------------------------------

    def subspace_iter(self, Y0):
        """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

        Uses a numerically stable subspace iteration algorithm to down-weight
        smaller singular values.

        :param A:       (m x n) matrix.
        :param Y0:      Initial approximate range of A.
        :param n_iters: Number of subspace iterations.
        :return:        Orthonormalized approximate range of A after power
                        iterations.
        """
        Q = self.ortho_basis(Y0)
        for _ in range(self.n_subspace_iters):
            Z = self.ortho_basis(self.AT(Q))
            Q = self.ortho_basis(self.A(Z))
        return Q

    # ------------------------------------------------------------------------------

    def ortho_basis(self, M):
        """Computes an orthonormal basis for a matrix.

        :param M: (m x n) matrix.
        :return:  An orthonormal basis for M.
        """
        Q, _ = np.linalg.qr(M)
        return Q
