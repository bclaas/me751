import numpy as np
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field
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
    _rdot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _pdot: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # TODO: Account for markers outside CG