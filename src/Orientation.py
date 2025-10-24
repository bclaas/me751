import numpy as np
from typing import Callable, Dict, Optional, Tuple

def vec2quat(theta: float, u: np.ndarray) -> np.ndarray:
    """ 
    Vector to quaternion

    theta (float): Angle of rotation, in radians.
    u (np.ndarray): Axis of rotation.
    """
    u = u / np.linalg.norm(u)
    e0 = np.cos(theta/2)
    [e1, e2, e3] = np.sin(theta/2) * u
    return np.array([e0, e1, e2, e3])

def tilde(v: np.ndarray):
    """
    Skew-symmetric matrix generator of a vector v
    """
    [x,y,z] = v
    return np.array([[0,-z, y],
                     [z, 0,-x],
                     [-y, x, 0]], float)

class Orientation:
    def __init__(self, e0, e1, e2, e3):
        """
        e0, e1, e2, e3: Euler parameters
        """
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
    
    @property
    def p(self):
        return [self.e0, self.e1, self.e2, self.e3]
    
    @property
    def A(self):
        [e0, e1, e2, e3] = self.p
        return 2*np.array([
                            [e0**2 + e1**2 - 0.5, e1*e2 - e0*e3, e1*e3 - e0*e2],
                            [e1*e2 + e0*e3, e0**2 + e2**2 - 0.5, e2*e3 - e0*e1],
                            [e1*e3 - e0*e2, e2*e3 + e0*e1, e0**2 + e3**2 - 0.5]
                        ], dtype=float)
    
    @property
    def E(self):
        [e0, e1, e2, e3] = self.p
        return np.array([
            [-e1,  e0,  -e3,  e2],
            [-e2,  e3,   e0, -e1],
            [-e3, -e2,   e1,  e0]
        ], dtype=float)

    @property
    def G(self):
        [e0, e1, e2, e3] = self.p
        return np.array([
            [-e1,  e0,   e3, -e2],
            [-e2, -e3,   e0,  e1],
            [-e3,  e2,  -e1,  e0]
        ], dtype=float)
    
    def set_p(self, p_new: np.ndarray):
        [self.e0, self.e1, self.e2, self.e3] = p_new
