from ngsolve import TaskManager, Mesh
from netgen.occ import Circle, MoveTo, Glue, OCCGeometry

from copy import deepcopy

class create_paper_geometry:
    def __init__(self, PML_extra_radius = 1, outer_radius = 1, mid_radius = 0.4, inner_radius = 0.35,  
                       n_scatterers = 3, PML = False,
                       PML_name = "pml", outer_name = "outer", mid_name = "mid", inner_name = "inner",
                       face_name = "face", scatterer_name = "scatterer"):
        
        self.PML_extra_radius = PML_extra_radius
        self.outer_radius = outer_radius
        self.mid_radius = mid_radius
        self.inner_radius = inner_radius
        
        self.n_scatterers = n_scatterers
        self.PML = PML

        self.PML_name = PML_name
        self.outer_name = outer_name
        self.mid_name = mid_name
        self.inner_name = inner_name
        self.face_name = face_name
        self.scatterer_name = scatterer_name

        self.PML_face_name = self.PML_name + self.face_name
        self.outer_face_name = self.outer_name + self.face_name
        self.mid_face_name = self.mid_name + self.face_name
        self.inner_face_name = self.inner_name + self.face_name

        self.names_dict = {self.PML_name: 0,
                           self.outer_name: 0,
                           self.mid_name: 0,
                           self.inner_name: 0,
                           self.scatterer_name: 0}
        
        self.min_sensor_radius = mid_radius * 1.1
        self.max_sensor_radius = outer_radius * 0.9
        
        assert inner_radius < mid_radius, "Inner radius must be less than mid radius!"
        assert mid_radius < outer_radius, "Mid radius must be less than outer radius!"
        if PML:
            assert 0 < PML_extra_radius, "PML extra radius must be greater than 0!"
        assert isinstance(self.n_scatterers, int) and self.n_scatterers > 0, "Must specify a positive integer amount of scatterers!"

        self.create_paper_geometry()
        
    # Recreates the geometry used in 
    # C. Aarset: Global optimality conditions for sensor placement, with extensions to binary low-rank A-optimal designs
    # https://doi.org/10.1088/1361-6420/add9bf
    def create_paper_geometry(self):
        with TaskManager():

            ###################
            ### Create geo ###
            ###################
            
            self.inner = Circle((0.0, 0.0), self.inner_radius).Face()
            self.inner.faces.name = self.inner_name + self.face_name
            self.inner.edges.name = self.inner_name
            
            self.outer = Circle((0.0, 0.0), self.outer_radius).Face()
            self.outer.edges.name = self.outer_name

            self.mid = Circle((0.0, 0.0), self.mid_radius).Face()

            self.pmlregion = Circle((0.0, 0.0), self.outer_radius + 0.2).Face()
            self.pmlregion.faces.name = self.PML_name
            self.pmlregion = self.pmlregion - self.outer

            self.outer = self.outer - self.mid
            self.mid = self.mid - self.inner
            self.mid.faces.name = self.mid_name + self.face_name
            self.mid.edges.name = self.mid_name

            scatterers = []

            for i in range(self.n_scatterers):
                if i==0:
                    scatterer = MoveTo(0.5, -0.2).Rectangle(0.1, 0.4).Face()
                if i==1:
                    scatterer = MoveTo(-0.6, -0.2).Rectangle(0.1, 0.5).Face()
                if i==2:
                    scatterer = MoveTo(-0.3, -0.75).Rectangle(0.5, 0.2).Face()
                scatterer.edges.name = self.scatterer_name + str(i)
                scatterers.append(scatterer)

            for scatter in scatterers:
                self.outer = self.outer - scatter
            self.outer.faces.name = self.outer_name + self.face_name

            self.bigouter = self.pmlregion + self.outer
            self.bigouter.faces.name = self.PML_name + self.outer_name + self.face_name

            if self.PML:
                self.geo = Glue([self.inner, self.mid, self.outer, self.pmlregion])
            else:
                self.geo = Glue([self.inner, self.mid, self.outer])

    def create_mesh(self, maxh = 0.03):
        self.mesh = Mesh(OCCGeometry(self.geo, dim=2).GenerateMesh(maxh = maxh))
        if self.PML:
            self.mesh.SetPML(pml.Radial(rad= self.PML_extra_radius, alpha=1j, origin=(0,0)), self.PML_name)

    def check_inner(self, coords):
        coords.reshape(-1,2)
        names_dict = deepcopy(self.names_dict)
        names_dict[self.inner_name] = 1

        # Create subdomain indicator using MaterialCF
        inner_indicator = self.mesh.MaterialCF(names_dict)
        return inner_indicator(self.mesh(coords[:, 0], coords[:, 1]))
    
    def check_mid(self, coords):
        coords.reshape(-1,2)
        names_dict = deepcopy(self.names_dict)
        names_dict[self.mid_name] = 1

        # Create subdomain indicator using MaterialCF
        mid_indicator = self.mesh.MaterialCF(names_dict)
        return mid_indicator(self.mesh(coords[:, 0], coords[:, 1]))

    def check_outer(self, coords):
        coords.reshape(-1,2)
        names_dict = deepcopy(self.names_dict)
        names_dict[self.outer_name] = 1

        # Create subdomain indicator using MaterialCF
        outer_indicator = self.mesh.MaterialCF(names_dict)
        return outer_indicator(self.mesh(coords[:, 0], coords[:, 1]))