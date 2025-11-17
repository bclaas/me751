import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union

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

def vecs2ori(f, g, h):
    A = np.zeros((3, 3))
    A[:, 0] = f
    A[:, 1] = g
    A[:, 2] = h
    e0, e1, e2, e3 = A_to_p(A)
    return Orientation(e0, e1, e2, e3)

def tilde(v: np.ndarray):
    """
    Skew-symmetric matrix generator of a vector v
    """
    [x,y,z] = v
    return np.array([[0,-z, y],
                     [z, 0,-x],
                     [-y, x, 0]], float)

def A_to_p(A: np.ndarray):

    # Project A to the nearest element of SO(3) to avoid drift
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:  # enforce right-handedness
        U[:, -1] *= -1
        R = U @ Vt

    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        e0 = 0.25 * S
        e1 = (R[2,1] - R[1,2]) / S
        e2 = (R[0,2] - R[2,0]) / S
        e3 = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            e0 = (R[2,1] - R[1,2]) / S
            e1 = 0.25 * S
            e2 = (R[0,1] + R[1,0]) / S
            e3 = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            e0 = (R[0,2] - R[2,0]) / S
            e1 = (R[0,1] + R[1,0]) / S
            e2 = 0.25 * S
            e3 = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            e0 = (R[1,0] - R[0,1]) / S
            e1 = (R[0,2] + R[2,0]) / S
            e2 = (R[1,2] + R[2,1]) / S
            e3 = 0.25 * S

    p = np.array([e0, e1, e2, e3], dtype=float)
    p /= np.linalg.norm(p)
    if p[0] < 0.0:  # fix overall sign (optional but handy for continuity)
        p = -p
    
    return p

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
        return np.array([self.e0, self.e1, self.e2, self.e3])
    
    @property
    def A(self):
        [e0, e1, e2, e3] = self.p
        return 2*np.array([
                            [e0**2 + e1**2 - 0.5, e1*e2 - e0*e3, e1*e3 + e0*e2],
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
        [self.e0, self.e1, self.e2, self.e3] = p_new / np.linalg.norm(p_new)
