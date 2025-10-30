import numpy as np
from .Assembly import Assembly
from .Bodies import RigidBody
from .KCons import KCon, DP1, DP2, D, CD

def run_positionAnalysis(asy: Assembly, dt: float, end_time: float, inner_iters=25):
    """Kinematic position analysis. Use Newton-Raphson to drive vector-valued function phi to 0."""
    q = np.zeros(7*asy.nb)
    for bdy in asy.bodies:
        q[7*bdy._id:7*bdy._id+3] = bdy.r
        q[7*bdy._id+3:7*bdy._id+7] = bdy.ori.p

    results = []

    t = 0.0
    while t < end_time:
        for _ in range(inner_iters):
            nq = asy.nq
            phi_base = asy.get_Phi(t)
            jac_base = asy.get_Phi_q(t)    # Jacobian == Phi_q matrix

            m0 = len(phi_base)
            m = m0 + asy.nb

            phi = np.zeros(m)
            phi[0:m0] = phi_base
            jac = np.zeros((m, m))
            jac[:m0, :nq] = jac_base

            # Add holonomic Euler Parameter normalization constraints
            for ii, bdy in enumerate(asy.bodies):
                p_bdy = bdy.ori.p
                phi[m0+ii] = 0.5*(p_bdy.T @ p_bdy - 1)

                row = np.zeros(nq)
                row[7*bdy._id+3:7*bdy._id+7] = p_bdy
                jac[m0+ii,:] = row
            
            # Dims of jac should be mxm at this point
            # Newton-Raphson
            #correction = np.linalg.solve(jac, phi)
            correction, *_ = np.linalg.lstsq(jac, phi, rcond=None)  # HACK
            q = q - correction

            for idx in range(asy.nb):
                new_r = q[7*idx:7*idx+3]
                new_p = q[7*idx+3:7*idx+7]
                asy.bodies[idx].r = new_r
                asy.bodies[idx].ori.set_p(new_p)

        results.append(np.append(q, t))

        t = t + dt
    
    return np.array(results)