# Base np/scipy imports
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, LinearOperator, eigsh, ArpackError
from scipy.linalg import cholesky, solve

# NGSolve import
import ngsolve as ngs
from ngsolve import H1, L2, Compress, TaskManager, BilinearForm, LinearForm, dx, ds, GridFunction, Integrate, IfPos
from ngsolve.la import BaseMatrix
from ngsolve.solvers import LOBPCG

# Useful tools import
import dill as pickle
from itertools import count
from copy import deepcopy
from time import time
from datetime import timedelta
from functools import partial

# File storage tools import
from os import listdir, makedirs
from os.path import isfile, join

# Own files import
from safe_add_CFs import safe_add_CFs, interp_add_CFs
from util import mat_to_csc, csc_to_chol, xfem_handler # Warning: Check CHOLMOD license.
from create_observation import create_observation

class mesh_maker():
    def __init__(self, 
                # Global rng for reproducibility
                rng,
                # Problem geometry passed as argument
                geo_creator,
                # Arguments related to the mesh
                PML = False, maxh = 0.05,  refine_inner = False, post_refine = 1,
                # Arguments relating to FEM
                order = 1, integration_order = 10, solver = 'pardiso',
                # Arguments related to prior
                alpha = 0.01125, beta = 1, robin = 1.42, prior_scale = 1,
                # Real-complex specification
                domain_is_complex = False, data_is_complex = True,
                # Grid-related
                target_m = 100, sensor_shape = "round", sensor_radius = 0, base = 24):
        
        ### Set selfs from init ###
        self.rng = rng

        self.geo_creator = geo_creator
        
        self.PML = PML
        self.maxh = maxh
        self.refine_inner = refine_inner
        self.post_refine = post_refine

        self.order = order
        self.integration_order = integration_order
        self.solver = solver

        self.alpha = alpha
        self.beta = beta
        self.robin = robin
        self.prior_scale = prior_scale

        self.domain_is_complex = domain_is_complex
        self.data_is_complex = data_is_complex

        self.target_m = target_m
        self.sensor_shape = sensor_shape
        self.sensor_radius = sensor_radius
        self.base = base

        self.meas0 = None
        
        ### Prepare storage ###
        self.PDE = {}
        self.APDE = {}
        self.MassConverters = {}
        self.Prior = None

        # Complex data is treated as double-length real vectors
        if self.data_is_complex:
            self.dtype = np.complex64
            self.out_map = lambda g: np.concatenate((g.real,g.imag))
            self.out_map_inv = lambda g: g[:len(g)//2] + 1j * g[len(g)//2:]
        else:
            self.dtype = np.float64
            self.out_map = lambda g: g.real
            self.out_map_inv = lambda g: g

        # Set to True once data has been successfully loaded
        self.load_flag = False

        # Attempt to load data
        self.load_data()

        # Build everything we need
        # Each call is aware of whether data has been loaded or not,
        # and some may output nothing if it has been loaded

        self.create_mesh()
        self.create_grid()
        self.create_fes()
        self.create_Prior()
        self.create_Omat()
            
        # Attempt to save data
        self.save_data()
            
        self.n = self.fes.ndof
        self.m_sensors = self.grid.shape[0]
        if self.data_is_complex:
            self.m_sensors_actual = self.m_sensors * 2
        else:
            self.m_sensors_actual = self.m_sensors
    
    # Loads data if it exists
    def load_data(self):
        self.mesh_dumps_directory = "ngs_dumps/"
        makedirs(self.mesh_dumps_directory, exist_ok = True)
        
        self.target_filename = "maxh_" + str(self.maxh) + "_scat_" + str(self.geo_creator.n_scatterers) + \
                          "_order_" + str(self.order) + "_refine_" + str(int(self.refine_inner)) + \
                          "_target_" + str(self.target_m) + "_shape_" + self.sensor_shape + "_radius_" + str(self.sensor_radius)
        print("-"*40)
        print("Associated filename:",self.target_filename)
        print("-"*40)
        for filename in listdir(self.mesh_dumps_directory):
            if isfile(join(self.mesh_dumps_directory, filename)):
                if self.target_filename == filename:
                    with open(join(self.mesh_dumps_directory, filename), "rb") as input_file:
                        obj = pickle.load(input_file)

                        self.mesh = obj["mesh"]

                        # Prior-related
                        self.diagPrior = obj["diagPrior"]
                        self.tracePrior = obj["tracePrior"]
                        
                        # Grid-related
                        self.grid = obj["grid"]
                        self.base = obj["base"]
                        self.Omat = obj["Omat"]
                        self.obstime = obj["obstime"]
                        self.circ_coords = obj["circ_coords"]
                        
                        self.load_flag = True
                        
                        print("Mesh successfully loaded.")
                        break

    # Saves data if it was not loaded this run
    def save_data(self, force = False):
        if self.load_flag and not force:
            return
        
        obj = {"mesh": self.mesh,
                "grid": self.grid, 
                "diagPrior": self.diagPrior, "tracePrior": self.tracePrior,
                "base": self.base,
                "Omat": self.Omat, "obstime": self.obstime,
                "circ_coords": self.circ_coords}
        
        with open(self.mesh_dumps_directory + self.target_filename, "wb") as output_file:
            pickle.dump(obj, output_file)    
        print("Successfully stored mesh.")

            
    # Create problem mesh
    def create_mesh(self, force = False):
        if self.load_flag and not force:
            pass
        else:
            print("No saved mesh found, generating...")
            self.geo_creator.create_mesh(maxh = self.maxh)
            self.mesh = self.geo_creator.mesh
            # Optionally refine inner part to emphasise source reconstruction.
            if self.refine_inner:
                for _ in range(int(self.refine_inner)):
                    print("Inner refining...")
                    for el in self.mesh.Elements():
                        self.mesh.SetRefinementFlag(el, el.mat == self.geo_creator.inner_face_name)
                    self.mesh.Refine()
        self.xfem_handler = xfem_handler(self.mesh)

    # Create feses
    def create_fes(self, create_Mass = False):

        self.fes = H1(self.mesh, order = self.order, 
                      complex = self.domain_is_complex, 
                      definedon = self.geo_creator.inner_face_name)
        self.fes = Compress(self.fes)
        
        self.cofes = H1(self.mesh, order = self.order, complex = self.data_is_complex)

        # n is the number of free source dofs.
        self.n = self.fes.ndof

        print("Dofs:",(self.fes.ndof,self.cofes.ndof))
        if create_Mass:
            print("Creating mixed mass matrices...")
            for fes in [self.fes, self.cofes]:
                self.create_Mass(fes)

    # Create observation grid
    def create_grid(self):
        
        if self.load_flag:
            pass
        else:
            self.sqrtm = int(np.log2(self.target_m/self.base + 1))

            # Loops until we've managed to pack at least target_m sensors in the grid,
            # while trying not to overshoot more than necessary
            while True:
                R = np.linspace(self.geo_creator.min_sensor_radius, self.geo_creator.max_sensor_radius, self.sqrtm)

                coords = np.array([])
                # Places sensors in rings of radius r \in R
                for i, r in enumerate(R):
                    count = self.base * 2**i
                    T = np.linspace(0, 2 * np.pi, int(count), endpoint = False)
                    
                    X = r * np.cos(T)
                    Y = r * np.sin(T)

                    if not i:
                        coords = np.stack((X.ravel(), Y.ravel()), axis = 1)
                    else:
                        coords = np.vstack((coords,np.stack((X.ravel(), Y.ravel()), axis = 1)))

                # Filter out sensors not actually in the mesh
                in_mesh = np.array([self.mesh.Contains(*coord) for coord in coords],dtype=bool)
                if np.sum(in_mesh) < self.target_m:
                    self.sqrtm += 1
                    continue
                else:
                    coords = coords[in_mesh,:]
                    del X, Y, in_mesh
                    break

            self.grid = coords
            self.circ_coords = self.grid[np.argsort(-np.linalg.norm(coords, axis = 1)),:]
            print("Finished building grid.")

        # Count the number of sensors.
        # For complex data, we treat it as though every sensor appears twice,
        # with two outputs "in the same spot" - 
        # one real, one complex, concatenated as a real double-length vector.
        self.m_sensors = self.grid.shape[0]
        
        # Create observation functions
        
        self.create_observation = create_observation(shape=self.sensor_shape, fallback_radius = 5e-2 / max(25, self.m_sensors))

        self.is_pointwise = self.create_observation.is_pointwise
        self.sensors_are_supported = self.create_observation.is_supported
        self.dim = self.create_observation.dim

        self.peak = self.create_observation.peak
        self.sensor_norm = self.create_observation.norm

        if self.sensor_radius:
            self.peak /= self.sensor_radius ** self.dim
            self.sensor_norm /= self.sensor_radius ** (self.dim / 2) 
        
        if not self.load_flag:
            # Adaptively refine the support of the sensors
            # This increases accuracy of observations at relatively negligible cost
            if self.sensors_are_supported:
                supps = partial(self.observation_functions, yield_obs = False, yield_supp = True)
                self.xfem_handler.refine_supports(supps = supps,
                                                 banned_regions = [self.geo_creator.inner_face_name,
                                                                   self.geo_creator.mid_face_name])
                
                # Must carry out a post-refine to avoid ngsolve bug
                # This sucks, please fix ngsolve team, thankx
                if self.post_refine:
                    for _ in range(int(self.post_refine)):
                        for el in self.mesh.Elements():
                            self.mesh.SetRefinementFlag(el, True)
                        self.mesh.Refine()

    def create_Omat(self):
    
        if self.load_flag:
            return
    
        # Create finite observation matrix
        self.Omat = lil_matrix((self.m_sensors,self.cofes.ndof))
        start_obstime = time()
        V = self.cofes.TestFunction()
        
        for k, tup in enumerate(self.observation_functions(yield_supp = True)):
            print("\rBuilding observation matrix: ",k,"/",self.m_sensors,"...",sep="",end="")
            obs, supp = tup
        
            # This gives precisely the FEM's interpretation of
            # pointwise evaluation by storing each basis element's
            # value in each sensor point. This can safely be stored
            # as a sparse matrix, since the (local) basis will only
            # have very few elements touching each sensor
            
            Lo = LinearForm(self.cofes)
            if self.is_pointwise:
                coord = supp
                Lo += V(*coord)
            else:
                if self.sensors_are_supported:
                    Lo += obs * V * self.xfem_handler.dCut_from_supp(supp = supp)
                else:
                    Lo += obs * V
            Lo.Assemble()
            Lo = Lo.vec.FV().NumPy()[:]
            assert np.all(Lo.imag == 0), "This approach does not work with complex-valued bases!"
            self.Omat[k,:] = Lo.real
            
        self.Omat = self.Omat.tocsc()
        
        self.obstime = time() - start_obstime
        print("\nBuilt observation matrix in",timedelta(seconds = self.obstime))

    # Observation functions defined as a generator for memory purposes
    def observation_functions(self, yield_obs = True, yield_supp = False):
        
        for coordx, coordy in self.grid:
            obs, supp = self.create_observation.create_observation(coordx = coordx, coordy = coordy, r = self.sensor_radius)

            out = []
            if yield_obs:
                out.append(obs)
            if yield_supp:
                out.append(supp)
            if len(out) == 1:
                yield out[0]
            else:
                yield tuple(out)

    def create_Prior(self):
        u, v = self.fes.TnT()

        self.PriorInv = BilinearForm(self.fes, symmetric = True)
        self.PriorInv += self.prior_scale * self.alpha * ngs.grad(u) * ngs.grad(v) * dx + self.prior_scale * self.beta * u * v * dx
        self.PriorInv += self.robin * u * v * ds
        
        self.PriorInv.Assemble()
        self.PriorInv = self.PriorInv.mat
        self.Prior = self.PriorInv.Inverse(freedofs = self.fes.FreeDofs(), inverse = self.solver)

        if not self.load_flag:
            print("Computing prior diag...")
            self.diagPrior = self.diag(Cov = self.C, power = 2)
            self.tracePrior = Integrate(self.diagPrior, self.mesh)
            print("Prior trace: ",self.tracePrior,"...",sep="")
            
           
    def create_PDE(self, omega):
        
        if omega in self.PDE.keys() and omega in self.APDE.keys():
            return
        
        # Helmholtz equation as PDE
        # Neumann boundary condition on scatterers implicitly imposed by not including it (=0)
        
        with TaskManager():
            U, V = self.cofes.TnT()

            PDE = BilinearForm(self.cofes)
            PDE += ngs.grad(U)*ngs.grad(V)*dx - omega**2*U*V*dx
            #a += -omega*1j*U*V * ds("pmlregion")
            PDE += -omega*1j*U*V * ds("outer")
            PDE.Assemble()

            PDE = PDE.mat.Inverse(freedofs = self.cofes.FreeDofs(), inverse=self.solver)

            # Adjoint Helmholtz equation as adjoint PDE
            APDE = BilinearForm(self.cofes)
            APDE += ngs.grad(U)*ngs.grad(V)*dx - omega**2*U*V*dx
            #a += -omega*1j*U*V * ds("pmlregion")
            APDE += omega*1j*U*V * ds("outer")
            APDE.Assemble()

            APDE = APDE.mat.Inverse(freedofs = self.cofes.FreeDofs(), inverse=self.solver)

        #norm = power_iteration(PDE = PDE, APDE = APDE, fes = self.cofes, iters = 100)
        #print(omega,"has norm",norm)

        self.PDE[omega] = PDE
        self.APDE[omega] = APDE
        
        return

    def diag(self, Cov, fes = None, order = None, power = 1, interp = False):

        real_to_ngs = self.real_to_ngs
        ngs_to_real = self.ngs_to_real
        
        if fes is None:
            try:
                fes = Cov(CF(1)).space
            except:
                fes = self.fes
        self.create_Mass(fes)

        if order == 0:
            return GridFunction(fes)
        
        n = fes.ndof
        if order is None:
            order = n-1

        if isinstance(Cov,BaseMatrix):

            def input_basevec(basevec):
                f = GridFunction(fes)
                f.vec[:] = basevec
                return f
                
            eigvals, eigvecs = LOBPCG(mata = Cov, matm = fes.MassMatrix, 
                                              pre = Cov.Inverse(freedofs = fes.FreeDofs(), inverse = self.solver), 
                                              num = order, maxit = self.max_eig_iters, printrates=True)
            diag_generator = (input_basevec(eigvec)**2 for eigvec in eigvecs)
        
        else:
            def realCov(r):
                f = real_to_ngs(r)
                f = Cov(f)
                return ngs_to_real(f)

            r = np.ones(n)
            CovOp = LinearOperator(matvec = realCov, rmatvec = realCov, shape = (n,n))
            try:
                eigvals, eigvecs = eigsh(CovOp, k = min(order,n-1))
            
            # If CovOp is the zero operator, this will error out; however, for us, that just means
            # that diag should be the zero function. This try/except fixes this, and returns all other
            # errors normally.
            except ArpackError as e:
                if e.__str__() == "ARPACK error -9: Starting vector is zero.":
                    return GridFunction(fes)
                raise
        diag_generator = (real_to_ngs(eigvec)**2 for eigvec in eigvecs.T)
            
        diagCF = safe_add_CFs(CFs = diag_generator, weights = iter(eigvals**power), length = order)
        if interp:
            diag = GridFunction(fes)
            diag.Set(diagCF)
            return diag
        else:
            return diagCF
    
    def A(self, f, omega = 50):
        
        self.create_PDE(omega)
        
        with TaskManager():
            V = self.cofes.TestFunction()

            u = GridFunction(self.cofes)
            Lf = LinearForm(f * V * dx).Assemble()
            u.vec.data += self.PDE[omega] * Lf.vec
            
        return u

    def AT(self, u, omega = 50):
        
        self.create_PDE(omega)
        
        with TaskManager():
            V = self.cofes.TestFunction()

            F = GridFunction(self.cofes)
            Lu = LinearForm(u * V * dx).Assemble()
            
            F.vec.data += self.APDE[omega] * Lu.vec
            
            f = GridFunction(self.fes)
            if self.domain_is_complex:
                f.Set(F)
            else:
                f.Set(np.real(F))
        return f
    
    def C(self, f):
        with TaskManager():
            v = self.fes.TestFunction()

            F = GridFunction(self.fes)
            F.vec.data = self.Prior * LinearForm(f * v * dx).Assemble().vec
        return F

    def CT(self, f):
        with TaskManager():
            v = self.fes.TestFunction()

            F = GridFunction(self.fes)
            F.vec.data = self.Prior * LinearForm(f * v * dx).Assemble().vec
        return F

    # Creates a visualisation of the grid
    # Not actually used for computations
    def grid_to_ngs(self, w, interp = False, fes = None):
        # Used to convert design vector w to ngs function for observation.
        kwargs = {
                    "CFs": self.observation_functions(yield_supp = self.sensors_are_supported), 
                    "weights": w,
                    "supported": self.sensors_are_supported
                 }
        if interp:
            if fes is None:
                fes = self.fes
            gf = interp_add_CFs(fes = fes, **kwargs)
            return gf
        else:
            cf = safe_add_CFs(**kwargs)
            return cf

    def Obs(self, u):
        shape = self.shape
        if self.is_pointwise:
            with TaskManager():
                g = u(self.mesh(self.grid[:,0],self.grid[:,1])).ravel()
        else:
            with TaskManager():
                if self.sensors_are_supported:
                    g = None
                else:
                    g = np.array( \
                        Integrate(tuple(obs * u for obs in self.observation_functions()), self.mesh, order = self.integration_order)).ravel()
        return g#self.out_map(g)
    
    def O(self, u):
        if isinstance(u, ngs.CoefficientFunction) and not isinstance(u, ngs.GridFunction):
            uv = GridFunction(self.cofes)
            uv.Set(u)
        else:
            uv = u
        with TaskManager():
            g = self.Omat@uv.vec.FV().NumPy()[:]
        return self.out_map(g)

    def OT(self, g):
        self.create_Mass(self.cofes)
        u0 = self.Omat.T.conj()@self.out_map_inv(g)
        u0 = self.cofes.Mchol["solve"](u0)
        return self.coeff_to_ngs(u0, fes = self.cofes)

    def F(self, f, omegas = [50]):
        g = np.array([])
        
        for omega in omegas:
            g = np.concatenate((g,self.O(self.A(f, omega = omega))))
        return g

    def FT(self, g, omegas = [50]):
        
        f = GridFunction(self.fes) 
        for k_obs, omega in enumerate(omegas):
            gr = np.roll(g,-k_obs * self.m_sensors_actual)
            gr = gr[:self.m_sensors_actual]
            f.vec.data += np.real(self.AT(self.OT(gr), omega = omega).vec.data)
        return f

    def FC(self, r, omegas = [50]):
        f = self.real_to_ngs(r)
        return self.F(self.C(f), omegas = omegas)

    def CFT(self, g, omegas = [50]):
        f = self.C(self.FT(g, omegas = omegas))
        return self.ngs_to_real(f)
    
    def trace(self, Cov, fes = None):
        # Computes the trace of any arbitrary
        # ngsolve operator
        coeff_to_ngs = self.coeff_to_ngs
        
        if fes is None:
            try:
                fes = Cov(CF(1)).space
            except:
                fes = self.fes
        
        n = self.n
        tr = 0

        # Unit vectors (Euclidean)
        def e(i):
            e = np.zeros(n)
            e[i] = 1
            return e

        with TaskManager():
            for i in range(n):
                print("\rComputing cov trace, index " + str(i) + "/" + str(n) + "...", sep="",end="")
                tr += self.ngs_to_real(Cov(self.real_to_ngs(e(i))))[i]
        print("")
        return tr
    
    def sample(self, Ch, fes = None):
        if fes is None:
            fes = self.fes
        eta = self.rng.normal(size = fes.ndof)
        eta = self.real_to_ngs(eta)
        sample = Ch(eta)
        return sample

    def create_Mass(self, fes1, fes2 = None):
        # Builds mass matrix and accompanying Cholesky 
        # decomposition if not already present
        # Can also build the converter between feses

        # Must decide if we need to build this
        if fes2 is None:
            fes2 = fes1

        # Workaround needed for complex-to-real conversion:
        # Map from real to complex and transpose.
        complex_to_real = fes1.is_complex and not fes2.is_complex

        def create(fes1, fes2, create_Mchol = True):
            u1 = fes1.TrialFunction()
            v2 = fes2.TestFunction()

            with TaskManager():
                MassMatrix = BilinearForm(trialspace=fes1, testspace=fes2)
                MassMatrix += u1 * v2 * dx
                MassMatrix.Assemble()
                MassMatrix = MassMatrix.mat

                if create_Mchol:
                    MassMatrixCSC = mat_to_csc(MassMatrix)
                    # For 0th order FEM, we throw away off-diagonals
                    if fes1.globalorder == 0:
                        MassMatrixCSC = diags(MassMatrixCSC.diagonal(), format = "csc")
                    Mchol = csc_to_chol(MassMatrixCSC)
                    return MassMatrix, Mchol
                else:
                    return mat_to_csc(MassMatrix)

        def inner(f,g,mesh=None):
            try:
                assert f.space == g.space, "Spaces do not match!"
                space = f.space
                Mg = GridFunction(space)
                Mg.vec.data = space.MassMatrix * g.vec
                return np.vdot(Mg.vec.FV().NumPy()[:], f.vec.FV().NumPy()[:]).real
            except:
                if mesh is None:
                    try:
                        mesh = f.space.mesh
                    except:
                        try:
                            mesh = g.space.mesh
                        except:
                            raise Exception("Either f or g must have a fes with an associated mesh, or the mesh argument must be provided!")
                return Integrate(f*Conj(g),mesh)

        def norm(f):
            return inner(f,f)**(1/2)
        
        def error(f,g):
            try:
                assert f.space == g.space, "Spaces do not match!"
                space = f.space
                err = GridFunction(space)
                err.vec.FV().NumPy()[:] = f.vec.FV().NumPy()[:] - g.vec.FV().NumPy()[:]
            except:
                err = f - g
            return norm(err)

        def relative_error(f,g):
            return error(f,g)/max(norm(f),norm(g))

        # Populate both feses with mass matrix and Cholesky decomp
        for fes in [fes1, fes2]:
            if not hasattr(fes, "MassMatrix"):
                fes.MassMatrix, fes.Mchol = create(fes1 = fes, fes2 = fes, create_Mchol = True)
                fes.MassMatrixCSC = mat_to_csc(fes.MassMatrix)
                fes.MassMatrixDiagonal = fes.MassMatrixCSC.diagonal()
                fes.inner = inner
                fes.norm = norm
                fes.error = error
                fes.relative_error = relative_error
            
        if fes1 != fes2:
            if id(fes1) not in self.MassConverters.keys():
                self.MassConverters[id(fes1)] = {}
            if id(fes2) not in self.MassConverters.keys():
                self.MassConverters[id(fes2)] = {}
                
            if id(fes2) not in self.MassConverters[id(fes1)].keys():
                #try:
                #    # More efficient to just use the Hermitian as the opposite-direction mass matrix
                #    self.MassConverters[id(fes1)][id(fes2)] = self.MassConverters[id(fes2)][id(fes1)].T
                #except:
                self.MassConverters[id(fes1)][id(fes2)] = create(fes1 = fes1, fes2 = fes2, create_Mchol = False)
                        
    def fes_to_fes(self, f, fes2):
        
        assert hasattr(f, "space"), "f must have a space to convert from!"
        fes1 = f.space
        
        if fes1 == fes2:
            print("Input and output feses are equal, check if this code needs to be called here...")
            return f
        else:
            self.create_Mass(fes1 = fes1, fes2 = fes2)
            partially_interpolated = GridFunction(fes2)
            interpolated = GridFunction(fes2)
            
            partially_interpolated.vec.FV().NumPy()[:] = self.MassConverters[id(fes1)][id(fes2)] @ f.vec.FV().NumPy()[:]
            interpolated.vec.data = fes2.MassMatrix.Inverse(freedofs = fes2.FreeDofs()) * partially_interpolated.vec
            return interpolated

    def coeff_to_ngs(self, r, fes = None):
        
        if fes is None:
            fes = self.fes
            
        with TaskManager():
            f = GridFunction(fes)
            if fes.is_complex:
                f.vec.FV().NumPy()[:] = r
            else:
                f.vec.FV().NumPy()[:] = r.real
        return f
        
    def ngs_to_coeff(self, f, fes = None):
        if fes is None:
            try:
                fes = f.space
            except:
                fes = self.fes
        
        # Allows sending CFs to coeffs
        if not hasattr(f, "space"):
            fs = GridFunction(fes)
            fs.Set(f)
            f = fs
        
        with TaskManager():
            if fes.is_complex:
                return f.vec.FV().NumPy()[:]#.conj()
            else:
                return f.vec.FV().NumPy()[:].real

    def real_to_ngs(self, r, fes = None, full_power = False):
        if fes is None:
            fes = self.fes
        self.create_Mass(fes)
        solve = fes.Mchol["solve_h"](r)
        if full_power:
            solve = fes.Mchol["solve_hT"](solve)
        return self.coeff_to_ngs(solve, fes = fes)

    def ngs_to_real(self, f, fes = None, full_power = False):
        if fes is None:
            try:
                fes = f.space
            except:
                fes = self.fes
        self.create_Mass(fes)
        if not full_power:
            apply = fes.Mchol["apply_h"]
        else:
            apply = fes.Mchol["apply"]
        return apply(self.ngs_to_coeff(f, fes = fes))