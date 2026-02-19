import autograd.numpy as np

def rowsum(A):
    return A.sum(1)

def diagg(A, B):
    return rowsum(A@B)

def khatri_rao(A, B, method = "column"):
    assert method.casefold() == "column" or method.casefold() == "row", "Argument 'method' must be one of 'column' and 'row' (case insensitive)!"
    if method.casefold() == "column":
        def khra(x):
            return (A@(x*B).T).ravel()
    else:
        def khra(x):
            return (B@x.reshape((B.shape[1],-1))@A.T).diagonal()

    return khra

def kroenecker(A, B = None):
    if B is None:
        B = A
    n, m = B.shape

    # Input-aware: Uses fast slicing operations to
    # rapidly apply kroenecker product to matrix input
    # whenever necessary, otherwise defaults to 
    # standard matvec formula
    def kron(x):
        if np.ndim(x) == 1:
            return (B@x.reshape((B.shape[1],-1))@A.T).ravel()
        else:
            _, l = x.shape
            Y = x.reshape((m,-1),order="F")
            Y = B @ Y
            Y = np.vstack(np.hsplit(Y,l))
            Y = Y @ A.T
            Y = np.ravel(np.vsplit(Y,l),order="F").reshape((-1,l))
            return Y
    return kron

def schur(A, B):
    def schr(x):
        return diagg(A, (x*B).T)

    return schr