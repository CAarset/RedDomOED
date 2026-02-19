from ngsolve import H1, GridFunction, TaskManager, Conj, LinearForm, dx, VOL, IfPos
import numpy as np
from sksparse.cholmod import cholesky
from scipy.sparse import csc_matrix
from xfem import Integrate, SymbolicLFI, POS, Integrate
from xfem.lsetcurv import LevelSetMeshAdaptation, dCut, InterpolateToP1, RefineAtLevelSet

def ngs_to_max(ngsf, mesh = None, definedon = None):
    with TaskManager():
        if mesh is None:
            mesh = ngsf.space.mesh
        if definedon is None:
            fes = H1(mesh, order = 0)
        else:
            fes = H1(mesh, order = 0, definedon = definedon)
        gf = GridFunction(fes)
        gf.Set(ngsf)
    return np.max(np.abs(gf.vec.FV().NumPy()[:]))

def mat_to_csc(mat, real = False):
    rows,cols,vals = mat.COO()
    if real:
        vals = vals.NumPy()[:].real
    return csc_matrix((vals, (rows, cols)), shape=(mat.height, mat.width))
    
def csc_to_chol(csc):
    factor = cholesky(csc)
    L = factor.L()
    
    chol = {}
    
    def handle_complex(f):
        def hcf(x):
            if np.any(np.iscomplex(x)):
                return f(x.real) + 1j * f(x.imag)
            return f(x.real)
        return hcf
    
    chol["apply"] = handle_complex(lambda x: csc@x)
    chol["apply_T"] = handle_complex(lambda x: csc.T@x)
    
    chol["apply_h"] = handle_complex(lambda x: L.T@factor.apply_P(x))
    chol["apply_hT"] = handle_complex(lambda x: factor.apply_Pt(L@x))
    
    chol["solve_h"] = handle_complex(lambda x: factor.apply_Pt(factor.solve_Lt(x, use_LDLt_decomposition = False)))
    chol["solve_hT"] = handle_complex(lambda x: factor.solve_L(factor.apply_P(x), use_LDLt_decomposition = False))
    
    chol["solve"] = handle_complex(lambda x: factor(x))
    chol["solve_T"] = lambda x: Exception("Not implemented.")
    
    chol["L"] = L
    chol["LT"] = L.T

    chol["hT"] = factor.apply_Pt(L)
    chol["h"] = chol["hT"].T
    
    return chol

def mat_to_eigs(mat, fes):

    U, V = fes.TnT()

    M = BilinearForm(fes)
    M += U * V * dx
    M.Assemble()

    mpre = BilinearForm(cofes)
    mpre += U * V * dx
    #mpre.mat = mpre.m + mat
    
    pre = Preconditioner(mpre, "direct", inverse="sparsecholesky")
    mpre.Assemble()

    evals, _ = solvers.PINVIT(Lap.mat, M.mat, pre = pre, num=max(2*len(omegas),12), maxit=20, printrates = False)
    return evals.NumPy()

def power_iteration(PDE, APDE, fes, iters = 10):
    u = GridFunction(fes)
    u.vec.FV().NumPy()[:] = np.random.normal(fes.ndof)
    try:
        u.vec.FV().NumPy()[:] += 1j * np.random.normal(fes.ndof)
    except:
        pass

    v = GridFunction(fes)
    w = GridFunction(fes)
    
    pair = lambda u, v: np.real(Integrate(u*Conj(u),fes.mesh))
    norm = lambda u: np.sqrt(pair(u,u))
    V = fes.TestFunction()
    
    for _ in range(iters):
        v.vec.data = PDE * LinearForm(u * V * dx).Assemble().vec
        w.vec.data = APDE * LinearForm(v * V * dx).Assemble().vec
        
        w_norm = norm(w)
        u.vec.data = w.vec.data / w_norm

    v.vec.data = PDE * LinearForm(u * V * dx).Assemble().vec
    w.vec.data = APDE * LinearForm(v * V * dx).Assemble().vec
    return pair(u, w) / pair(u, u)

def power_iteration_matrix(A, n, iters = 10):
    u = np.random.normal(size=n)
    
    pair = lambda u, v: np.real(np.inner(u,v))
    norm = lambda u: np.sqrt(pair(u,u))
    
    for _ in range(iters):
        v = A(u)
        v_norm = norm(v)
        u = v / v_norm

    return pair(u, A(u)) / pair(u, u)

##############################
### xfem utility functions ###
##############################

# Creates an xfem differential form suitable for integrating functions over (arbitrary small) supports defined via level sets
# Suitable for usage in LinearForms, replacing the usual dx
# Default values are set rather high

class xfem_handler:
    def __init__(self, mesh, threshold = 1, supp_direction = POS, order_int = 50, order_geom = 8, subdivlvl = 0):
        self.mesh = mesh
        self.threshold = threshold
        self.supp_direction = supp_direction
        self.order_int = order_int
        self.order_geom = order_geom
        self.subdivlvl = subdivlvl

    def levelset_from_supp(self, supp):
        return {"levelset": supp,
                "domain_type": self.supp_direction,
                "subdivlvl": self.subdivlvl,
                "order": self.order_int}

    def Integrate(self, cf, supp):
        #dC = self.dCut_from_supp(supp = supp, supp_max = supp_max)
        #return Integrate(cf * dC, mesh = self.mesh)
        meas = Integrate(levelset_domain = self.levelset_from_supp(supp), mesh = self.mesh, cf = cf)
        # Fallback in case of imprecise integration
        if meas == 0:
            meas = Integrate(IfPos(supp,cf,0),self.mesh,order=self.order_int)
        return meas

    def meshadaptation_from_supp(self, supp, supp_max = None, calc_deformation = True):
        # Create P1 approximation of level set
        gf_supp = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(supp, gf_supp)
        
        # Automatically computes max of P1 approximation of level set
        if supp_max is None:
            supp_max = ngs_to_max(gf_supp)

        # Apply isoparametric mapping for higher order geometry accuracy
        lsetmeshadap = LevelSetMeshAdaptation(self.mesh, order=self.order_geom, 
                                  threshold = self.threshold * supp_max, 
                                  discontinuous_qn = True)
        if calc_deformation:
            deformation = lsetmeshadap.CalcDeformation(supp)
            return lsetmeshadap, deformation
        return lsetmeshadap
    
    def dCut_from_supp(self, supp, supp_max = None):
        lsetmeshadap, deformation = self.meshadaptation_from_supp(supp = supp, supp_max = supp_max)
        return dCut(levelset = lsetmeshadap.lset_p1, 
                    domain_type = self.supp_direction, 
                    order = self.order_int, 
                    deformation = deformation,
                    subdivlvl = self.subdivlvl)
    
    def refine_supports(self, supps, tol = 0, target_nels = 15, banned_regions = None):
        
        if isinstance(banned_regions,str):
            banned_regions = [banned_regions]

        while True:

            total_indicators = np.zeros(self.mesh.ne)
            total_indicators = total_indicators.astype(bool)
            current_nels = []

            for supp in supps():
                
                indicator = IfPos(supp,1,0)
                element_indicators = np.array(Integrate(indicator, self.mesh, VOL, element_wise=True, order = self.order_int)).ravel()
                element_indicators[np.isinf(element_indicators)] = 1
                element_indicators[np.isnan(element_indicators)] = 0
                element_indicators[element_indicators<tol] = 0
                element_indicators = element_indicators.astype(bool)

                marked_nels = element_indicators.sum()

                # Drop elements in banned regions
                # Intentionally done after counting marked nels
                # to prevent over-refining on a small part outside of
                # the banned regions
                if banned_regions is not None:
                    for banned_region in banned_regions:
                        for k, el in enumerate(self.mesh.Elements()):
                            if el.mat == banned_region:
                                element_indicators[k] = 0

                # Stop contributing to refinement if we already have enough elements
                if marked_nels < target_nels:
                    total_indicators = np.logical_or(total_indicators, element_indicators)
                current_nels.append(marked_nels)
            
            print("Total nels:",self.mesh.ne)
            print("Current nels:",np.array(current_nels))
            # Stop if all supps are done refining
            if total_indicators.sum() == 0:
                break

            for el, el_ind in zip(self.mesh.Elements(),total_indicators):
                self.mesh.SetRefinementFlag(el, el_ind)
            self.mesh.Refine()
        
        # Reset all flags to True for compatibility
        for el in self.mesh.Elements():
            self.mesh.SetRefinementFlag(el, True)
        

    def refine_support(self, supp, supp_max = None, lower = 0, upper = 0, banned_regions = None):
        lsetmeshadap, _ = self.meshadaptation_from_supp(supp = supp, supp_max = supp_max, calc_deformation = True)
        
        # refine cut elements:
        RefineAtLevelSet(gf = lsetmeshadap.lset_p1, lower = lower, upper = upper)
        if banned_regions is not None:
            for banned_region in banned_regions:
                for el in self.mesh.Elements():
                    if el.mat == banned_region:
                        self.mesh.SetRefinementFlag(el, False)
        self.mesh.Refine()