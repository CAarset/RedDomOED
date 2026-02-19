from ngsolve import x, y, exp, sqrt, Mesh, Integrate, IfPos
from netgen.occ import Circle, OCCGeometry

class mollifiers:
    def __init__(self, shape="round", dim = 2, pow = 1):

        self.shape = shape
        self.dim = dim
        self.pow = pow

        self.scale = 1
        self.peak = 1
        self.norm = 1

        if self.test_shape("pointwise"):
            self.moll = lambda x, y: 1
            self.supp = None
            return

        self.support_radius = 1
        r2 = lambda x, y: x**2 + y**2

        if self.test_shape("round"):
            self.moll = lambda x, y: 1 - r2(x,y) ** self.pow
            self.supp = self.moll
        elif self.test_shape("gauss"):
            self.moll = lambda x, y: exp(1 - 1/(1 - r2(x,y) ** self.pow))
            self.supp = lambda x, y: 1 - r2(x,y) ** self.pow
        elif shape.casefold() == "hat".casefold():
            self.moll = lambda x, y: 1
            self.supp = lambda x, y: 1 - r2(x,y) ** self.pow
        else:
            raise ValueError("Unknown sensor shape",shape,"accepted values are round, gauss and hat (case insensitive).")
        
        # Normalisation for mollifier definition
        # Create a one-time-use custom mesh for this
        geo = Circle(c = (0,0), r = self.support_radius).Face()
        mesh = Mesh(OCCGeometry(geo, dim = self.dim).GenerateMesh(maxh = 0.01))

        self.scale = Integrate(IfPos(self.supp(x,y),self.moll(x,y),0),mesh,order=50)
        self.norm = Integrate(IfPos(self.supp(x,y),self.moll(x,y)**2,0),mesh,order=50)**(1/2)

        del mesh, geo

        self.peak /= self.scale

    def test_shape(self, shape):
        return self.shape.casefold() == shape.casefold()

class create_observation:
    def __init__(self, shape="pointwise", dim = 2, pow = 1, fallback_radius = 1):
        
        self.shape = shape
        self.dim = dim
        self.pow = pow
        
        self.is_pointwise = self.test_shape("pointwise")
        self.is_supported = not self.is_pointwise

        self.mollifier = mollifiers(shape = self.shape, dim = self.dim, pow = self.pow)

        if self.test_shape(shape = "pointwise"):
            r2 = lambda x, y: x**2 + y**2

            self.moll = lambda x, y: exp(-1 / fallback_radius * r2(x,y))
            self.moll_supp = None
            
        self.moll = self.mollifier.moll
        self.moll_supp = self.mollifier.supp

        self.scale = self.mollifier.scale
        self.peak = self.mollifier.peak
        self.norm = self.mollifier.norm

    def create_observation(self, coordx, coordy, r = 1):
        xshift = (x - coordx)/r
        yshift = (y - coordy)/r

        obs = self.moll(x = xshift, y = yshift) / self.scale * 1 / r ** self.dim
        if self.moll_supp is None:
            supp = (coordx, coordy)
        else:
            supp = self.moll_supp(x = xshift, y = yshift)
        return obs, supp
    
    def test_shape(self, shape):
        return self.shape.casefold() == shape.casefold()
