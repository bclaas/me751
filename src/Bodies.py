import numpy as np
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from .Orientation import Orientation, vec2quat, tilde

@dataclass
class RigidBody:
    name: str
    id: int
    r: np.ndarray   # (x,y,z) of CG in G-RF
    ori: Orientation
    # TODO: Account for markers outside CG