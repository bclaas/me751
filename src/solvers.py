import numpy as np
from .Assembly import Assembly
from .Bodies import RigidBody
from .KCons import KCon, DP1, DP2, D, CD

def run_positionAnalysis(asy: Assembly, dt: float, end_time: float):
    """Kinematic position analysis. Use Newton-Raphson to drive vector-valued function asy.Phi to 0."""
    phi_base = asy.get_Phi(0.0)
    jac_base = asy.get_Phi_q(0.0)    # Jacobian == Phi_q matrix

    n0 = len(phi)
    n = n0 + asy.nb

    phi = np.zeros(n)
    phi[0:n0] = phi_base
    jac = np.zeros((n, n))
    jac[0:n0,0:n0] = jac_base

    # Add Euler Parameter normalization constraints
    for ii, bdy in enumerate(asy.bodies):
        p_bdy = bdy.ori.p
        phi[n0+ii-1] = 0.5*(p_bdy.T @ p_bdy - 1)

        row = np.zeros(n)
        row[7*bdy._id+3:7*bdy._id+7] = p_bdy
        jac[n0+ii-1,:] = row