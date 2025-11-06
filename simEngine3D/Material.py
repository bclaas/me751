import numpy as np

class Material:
    def __init__(self, density: float, youngs: float, poissons: float, model: str="SVK"):
        
        if "venant" in model.lower() or model.lower() in ["svk", "sv"]:
            self._is_svk = True
            self._is_mr = False
        elif "mooney" in model.lower() or "rivlin" in model.lower() or model.lower() in ["mr", "m-r"]:
            self._is_svk = False
            self._is_mr = True
        else:
            raise ValueError(f"{model} is not a recognized material model. Try 'SVK' for St. Venant-Kirchoff, or 'M-R' for Mooney-Rivlin.")
        
        self.density = density
        self.youngs = youngs
        self.poissons = poissons
        
        