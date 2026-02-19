import numpy as np
import scipy

def lrQR(A, tol = 1e-20, target_rank = None, productmap = None, previousR = None, previousQ = None, previousdims = None, pivoting = False):

    if not pivoting:
        Q, R = scipy.linalg.qr(A, mode = 'economic')
        P = None
    else:
        Q, R, P = scipy.linalg.qr(A, mode = 'economic', pivoting = pivoting)
    
    print("QR done.")
    dR = np.diag(R)
    
    keep = np.argwhere(np.abs(dR) > tol * np.abs(np.max(dR)))
    keep = np.ravel(keep)
    if target_rank is not None:
        if target_rank < len(dR):
            keep = np.intersect1d(keep, \
                                   np.ravel(np.argpartition(dR, -target_rank)[-target_rank:]))
            keep = np.ravel(keep)
    R = R[keep,:]
    Q = Q[:,keep]

    newdim = len(keep)
    print("Dimension reduced from",A.shape,"to",newdim)
    return Q, R, P, newdim
