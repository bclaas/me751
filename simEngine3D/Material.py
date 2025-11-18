import numpy as np

# Helper function: Return material from one function
#def get_material(model, density, youngs, poissons):
    # if "venant" in model.lower() or model.lower() in ["svk", "sv"]:
    #     self._is_svk = True
    #     self._is_mr = False
    # elif "mooney" in model.lower() or "rivlin" in model.lower() or model.lower() in ["mr", "m-r"]:
    #     self._is_svk = False
    #     self._is_mr = True
    # else:
    #     raise ValueError(f"{model} is not a recognized material model. Try 'SVK' for St. Venant-Kirchoff, or 'M-R' for Mooney-Rivlin.")
    
    # TODO: the rest

# Material-model-independent tensors
def cauchy_green_strain(F: np.ndarray):
    # RIGHT Cauchy-Green Strain Tensor
    # Often denoted C
    # a.k.a. deformation tensor
    return F.T @ F

def green_lagrange_strain(F: np.ndarray):
    # Often denoted E
    # Symmetric
    # True strain measure; not affected by rigid body rotation; vanishes if undeformed
    return 0.5*(cauchy_green_strain(F) - np.eye(len(F)))

class Material:
    def __init__(self, density: float):
        self.density = density

class SVK(Material):
    # Saint-Venant-Kirchhoff
    # Materially (i.e. in terms of material properties) linear; geometrically non-linear
    def __init__(self, density, youngs, poissons):
        super().__init__(density)
        self.lmbda = youngs*poissons/((1+poissons)*(1-2*poissons))
        self.mu = youngs / (2*(1+poissons))
    
    def strain_energy_density(self, F):
        E = green_lagrange_strain(F)
        trE = np.trace(E)
        return 0.5*self.lmbda*(trE**2) + self.mu*np.trace(E @ E)
    
    def PK1(self, F):
        # First Piola-Kirchhoff stress tensor. Often denoted P.
        S = self.PK2(F)
        return F @ S
    
    def PK2(self, F):
        # Second Piola-Kirchhoff stress tensor. Often denoted S.
        E = green_lagrange_strain(F)
        return self.lmbda*np.trace(E)*np.eye(3) + 2*self.mu*E


class MooneyRivlin(Material):
    # Materially and Geometrically non-linear. Stress dependent on shape (as opposed to volume).
    def __init__(self):
        pass

class NeoHookeanMR(MooneyRivlin):
    # Simplest rubber-like model that gets some nonlinear deformation right
    def __init__(self):
        pass

class IncompressibleMR(MooneyRivlin):
    def __init__(self):
        pass

class Yeoh(Material):
    # Energy function is polynomial function of invariant I1bar
    # Drops invariant I2bar
    # Good for fitting experimental data for uniaxial stretch
    def __init__(self):
        raise NotImplementedError

class Ogden(Material):
    # Most complicated model
    # Based on principal stretches. Not reliant on invariants.
    def __init__(self):
        raise NotImplementedError
    

        
        