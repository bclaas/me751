import numpy as np
from typing import Callable, Dict, Optional, Tuple


def E_of_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p).reshape(4)
    e0, e1, e2, e3 = p
    return np.array([
        [-e1,  e0,  -e3,  e2],
        [-e2,  e3,   e0, -e1],
        [-e3, -e2,   e1,  e0]
    ], dtype=float)

def G_of_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p).reshape(4)
    e0, e1, e2, e3 = p
    return np.array([
        [-e1,  e0,   e3, -e2],
        [-e2, -e3,   e0,  e1],
        [-e3,  e2,  -e1,  e0]
    ], dtype=float)

def A_of_p(p: np.ndarray) -> np.ndarray:
    E = E_of_p(p)
    G = G_of_p(p)
    return E @ G.T

def omega_of(p: np.ndarray, pdot: np.ndarray) -> np.ndarray:
    return 2.0 * (E_of_p(p) @ np.asarray(pdot).reshape(4))

def _unitize_quat(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p).reshape(4)
    n = np.linalg.norm(p)
    if n == 0:
        raise ValueError("Quaternion has zero norm.")
    return p / n

def dp1(
    p_i: np.ndarray,
    p_j: Optional[np.ndarray],
    pdot_i: np.ndarray,
    pdot_j: Optional[np.ndarray],
    a_bar_i: np.ndarray,
    a_bar_j: np.ndarray,
    t: float = 0.0,
    f: Optional[Callable[[float], float]] = None,
    fdot: Optional[Callable[[float], float]] = None,
    fddot: Optional[Callable[[float], float]] = None,
    normalize_quats: bool = True
) -> Dict[str, np.ndarray]:
    f, fdot, fddot = (f or (lambda t: 0.0), fdot or (lambda t: 0.0), fddot or (lambda t: 0.0))

    p_i = _unitize_quat(p_i) if normalize_quats else np.asarray(p_i).reshape(4)
    A_i = A_of_p(p_i)
    a_i = A_i @ np.asarray(a_bar_i).reshape(3)

    if p_j is None:
        # Ground: a_j is given directly in G-RF via a_bar_j (user should pass G-RF value).
        A_j = np.eye(3)
        a_j = np.asarray(a_bar_j).reshape(3)
        pdot_j = np.zeros(4)
        p_j = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (unused for rotation here)
    else:
        p_j = _unitize_quat(p_j) if normalize_quats else np.asarray(p_j).reshape(4)
        A_j = A_of_p(p_j)
        a_j = A_j @ np.asarray(a_bar_j).reshape(3)

    # Constraint value
    Phi = float(a_i @ a_j - f(t))

    # Velocity RHS ν = ḟ(t)
    nu = float(fdot(t))

    # Jacobians w.r.t r (zero)
    Phi_r_i = np.zeros((1, 3))
    Phi_r_j = np.zeros((1, 3))

    # Jacobians w.r.t p
    cross_ij = np.cross(a_i, a_j)  # a_i × a_j
    Phi_p_i = (2.0 * cross_ij @ E_of_p(p_i)).reshape(1, 4)
    Phi_p_j = (-2.0 * cross_ij @ E_of_p(p_j)).reshape(1, 4)

    # Angular velocities
    omega_i = omega_of(p_i, pdot_i).reshape(3)
    omega_j = omega_of(p_j, pdot_j).reshape(3) if pdot_j is not None else np.zeros(3)

    # Helper cross combos
    wi_x_ai = np.cross(omega_i, a_i)
    wj_x_aj = np.cross(omega_j, a_j)

    term1 = np.cross(omega_i, np.cross(omega_i, a_i)).dot(a_j)
    term2 = 2.0 * wi_x_ai.dot(wj_x_aj)
    term3 = a_i.dot(np.cross(omega_j, np.cross(omega_j, a_j)))
    gamma = float(-(term1 + term2 + term3) + fddot(t))

    return {
        "Phi": np.array([Phi]),
        "nu": np.array([nu]),
        "gamma": np.array([gamma]),
        "Phi_r_i": Phi_r_i,
        "Phi_r_j": Phi_r_j,
        "Phi_p_i": Phi_p_i,
        "Phi_p_j": Phi_p_j,
        "a_i": a_i,
        "a_j": a_j,
        "omega_i": omega_i,
        "omega_j": omega_j
    }

def cd(
    r_i: np.ndarray,
    r_j: Optional[np.ndarray],
    v_i: np.ndarray,
    v_j: Optional[np.ndarray],
    p_i: np.ndarray,
    p_j: Optional[np.ndarray],
    pdot_i: np.ndarray,
    pdot_j: Optional[np.ndarray],
    s_bar_i: np.ndarray,
    s_bar_j: np.ndarray,
    c: np.ndarray,
    t: float = 0.0,
    f: Optional[Callable[[float], float]] = None,
    fdot: Optional[Callable[[float], float]] = None,
    fddot: Optional[Callable[[float], float]] = None,
    normalize_quats: bool = True
) -> Dict[str, np.ndarray]:
    
    f, fdot, fddot = (f or (lambda t: 0.0), fdot or (lambda t: 0.0), fddot or (lambda t: 0.0))

    r_i = np.asarray(r_i).reshape(3); v_i = np.asarray(v_i).reshape(3)
    p_i = _unitize_quat(p_i) if normalize_quats else np.asarray(p_i).reshape(4)
    A_i = A_of_p(p_i)
    s_i = A_i @ np.asarray(s_bar_i).reshape(3)

    if r_j is None:
        r_j = np.zeros(3); v_j = np.zeros(3)
        A_j = np.eye(3); s_j = np.asarray(s_bar_j).reshape(3)
        pdot_j = np.zeros(4); p_j = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        r_j = np.asarray(r_j).reshape(3); v_j = np.asarray(v_j).reshape(3)
        p_j = _unitize_quat(p_j) if normalize_quats else np.asarray(p_j).reshape(4)
        A_j = A_of_p(p_j)
        s_j = A_j @ np.asarray(s_bar_j).reshape(3)

    c = np.asarray(c).reshape(3)

    # Position-level
    Phi = float(c @ (r_j + s_j - r_i - s_i) - f(t))

    # Velocity RHS
    nu = float(fdot(t))

    # Jacobians
    Phi_r_i = (-c).reshape(1, 3)
    Phi_r_j = c.reshape(1, 3)
    Phi_p_i = (-2.0 * np.cross(c, s_i) @ E_of_p(p_i)).reshape(1, 4)
    Phi_p_j = ( 2.0 * np.cross(c, s_j) @ E_of_p(p_j)).reshape(1, 4)

    # Velocities
    omega_i = omega_of(p_i, pdot_i).reshape(3)
    omega_j = omega_of(p_j, pdot_j).reshape(3) if pdot_j is not None else np.zeros(3)

    # Acceleration RHS
    gamma = float(- c @ (np.cross(omega_j, np.cross(omega_j, s_j)) -
                         np.cross(omega_i, np.cross(omega_i, s_i))) + fddot(t))

    return {
        "Phi": np.array([Phi]),
        "nu": np.array([nu]),
        "gamma": np.array([gamma]),
        "Phi_r_i": Phi_r_i,
        "Phi_r_j": Phi_r_j,
        "Phi_p_i": Phi_p_i,
        "Phi_p_j": Phi_p_j,
        "s_i": s_i,
        "s_j": s_j,
        "omega_i": omega_i,
        "omega_j": omega_j
    }
