import numpy as np
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from .Orientation import Orientation, vec2quat, tilde

@dataclass
class RigidBody:
    name: str
    r: np.ndarray   # (x,y,z) of CG in G-RF
    ori: Orientation
    mass: float = None
    inertia: np.ndarray = None
    _id: int = None
    _is_ground: bool = False

    # TODO: Account for markers outside CG