import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .Orientation import Orientation, vec2quat, tilde
from .Material import Material


@dataclass
class Node:
    x: float
    y: float
    z: float
    nid: float  # Node ID


class Element:
    """General Element to inherit from"""

    def __init__(self, nids: List[int], mat: Material):
        self.node_ids = tuple(nids)
        self.mat = mat

    def dofs_per_node(self) -> int:
        raise NotImplementedError("Implement dofs_per_node()")

    def gauss_rule(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Implement gauss_rule()")

    def N(self, u: float, v: float, w: float) -> np.ndarray:
        raise NotImplementedError("Implement N(u,v,w)")

    def dN_dxi(self, u: float, v: float, w: float) -> np.ndarray:
        raise NotImplementedError("Implement dN_dxi(u,v,w)")

    def H(self, u: float, v: float, w: float, q_e: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement H(u,v,w,q_e)")


class B324(Element):
    """B3-24 beam"""
    def __init__(self, nids: List[Node], mat: Material, length: float, width: float, height: float):
        # Assumes nides == [p1, p2]
        assert len(nids) == 2
        super().__init__(nids, mat)
        self.length = length
        self.width = width
        self.height = height
        # TODO: Compute x-sectional areas and moments of inertia?
    
    @property
    def volume(self):
        return self.length * self.width * self.height
    
    @property
    def mass(self):
        return self.volume * self.mat.density

    def dofs_per_node(self) -> int: 
        return 12   # 4 nodal unknowns per node

    def gauss_rule(self):
        gps = np.array([(-1/np.sqrt(3), 0.0, 0.0),
                        ( 1/np.sqrt(3), 0.0, 0.0)])
        ws  = np.array([1.0, 1.0])
        return gps, ws

    def N(self, u, v, w):
        # TODO
        return np.zeros((1, 24))  # placeholder

    def dN_dxi(self, u, v, w):
        # TODO
        pass

    def H(self, u, v, w):
        """Jacobian of Shape Function"""
        rtn = np.zeros((8,3))
        rtn[0,0] = -3/(2*self.length) + (6*u**2)/(self.length**3)
        rtn[1,0] = -0.25 - u/self.length + (3*u**2)/(self.length**2)
        rtn[2,0] = -v / self.length
        rtn[2,1] = (0.5*self.length - u) / self.length
        rtn[3,0] = -w / self.length
        rtn[3,2] = (0.5*self.length - u) / self.length
        rtn[4,0] = 3/(2*self.length) - (6*u**2)/(self.length**3)
        rtn[5,0] = -0.25 + u/self.length + (3*u**2)/(self.length**2)
        rtn[6,0] = v / self.length
        rtn[6,1] = (0.5*self.length + u) / self.length
        rtn[7,0] = w / self.length
        rtn[7,2] = (0.5*self.length + u) / self.length
        return rtn

class ANCFBody:
    def __init__(self, name: str, r: np.ndarray, ori: Orientation):
        self.name = name
        self.r = r
        self.ori = Orientation  # Is this tracked for the body as a whole?
        self._id = None # TBD
        self._is_ground = False  # read-only
        self.elements = []
    
    @property
    def _is_ground(self):
        # read-only. ANCFBody can not be ground.
        return False