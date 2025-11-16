import time
import scipy
import numpy as np
from copy import deepcopy
from .Assembly import Assembly
from .Bodies import RigidBody
from .KCons import KCon, DP1, DP2, D, CD

def torque_from_lam(asy, lam_joint):
    Phiq_j = asy.get_Phi_q(0.0)
    Qc = Phiq_j.T @ lam_joint
    Qp = Qc[3:7]
    tau = 0.5*asy.bodies[0].ori.E @ Qp
    return tau

def _solveAxb(A, b, is_sym=False):
    try:
        if is_sym:
            x = scipy.linalg.solve(A, b, assume_a="sym", check_finite=False)
        else:
            x = scipy.linalg.solve(A, b, check_finite=False)
    except:
        x, *_ = np.linalg.lstsq(A, b)
    
    return x


def run_dynamics(asy, dt, end_time, write_increment=1, max_inner_its=10, relaxation=1.0, error_thres=np.inf):

    def get_matrix_eq(t):
        """Get matrix equation based on current state of asy"""
        problem_size = 8*asy.nb + asy.nc
        A = np.zeros((problem_size, problem_size))
        b = np.zeros(problem_size)
        mstart = 0                      # Start index for M matrix within A
        mend = 3*asy.nb                 # Last index for M matrix within A
        jpstart = mend                  # Start index for Jp matrix within A
        jpend = jpstart + 4*asy.nb      # Last index for Jp matrix within A

        # Build L2 matrix equation
        # | M_aug    Phi_aug^T  | * | qdotdot | = | f_aug   |
        # | Phi_aug  0          |   | lam     |   | gam_aug |

        # M_aug
        A[mstart:mend, mstart:mend] = asy.mass_matrix()
        A[jpstart:jpend, jpstart:jpend] = asy.inertia_matrix()

        # Phi_aug (build independently then add Phi_aug & Phi_aug.T to A)
        Phi_aug = np.zeros((asy.nb+asy.nc, 7*asy.nb))
        Phi_aug[0:asy.nb, mend:jpend] = asy.p_matrix()
        Phi_q = asy.get_Phi_q()
        Phi_aug[asy.nb:] = Phi_q
        A[jpend:, 0:jpend] = Phi_aug
        A[0:jpend, jpend:] = Phi_aug.T

        # Update vector b
        # Add generalized forces
        b[0:asy.nq] = asy.generalized_forces(t)

        # Add gamma^P to b
        for bdy in asy.bodies:
            b[asy.nq + bdy._id] = -2*(bdy._pdot.T @ bdy._pdot)  # NOTE: Drop the 2 `asy.p_matrix(x2=FALSE)`. Keep 2 if `asy.p_matrix(x2=TRUE)` (default)

        # Add gamma_hat to b
        for ii, jnt in enumerate(asy.joints):
            # TODO: Fix this if >1 ACE added per joint
            b[asy.nq + asy.nb + ii] = jnt.gamma(t)
        
        return A, b

    def get_jacobian(x0, t, a2v, v2q):
        # Store current state & reset later
        q_orig = asy.get_q()
        qdot_orig = asy.get_qdot()

        # Perturb the system of equations to get Jacobian
        m = len(x0)
        assert m == 8*asy.nb + asy.nc
        n = 8*asy.nb + asy.nc
        jac = np.empty((m,n), dtype=float)
        eps = np.sqrt(np.finfo(float).eps)

        def g_perturbed(idx, perturbation):
            x = x0[:]
            x[idx] += perturbation
            a_test = x[0:asy.nq]
            v_test = a2v(a_test)
            q_test = v2q(v_test)
            asy.set_q(q_test)
            asy.set_qdot(v_test)
            A, b = get_matrix_eq(t)
            return A @ x - b
        
        for idx in range(m):
            h = eps * max(1.0, abs(x0[idx]))
            gu = g_perturbed(idx, h)    # Correction when x_idx perturbed upward
            gd = g_perturbed(idx, -h)    # Correction when x_idx perturbed downward
            jac[:, idx] = (gu - gd)/(2*h)
        
        # Reset to input state
        asy.set_q(q_orig)
        asy.set_qdot(qdot_orig)

        return jac
    
    def _project_positions(q_in, t):
        # residual
        phi_base = asy.get_phi(t)        # f(t)=0 for your test
        phiP = []
        for b in asy.bodies:
            p = b.ori.p
            phiP.append(0.5*(p @ p - 1.0))
        varphi = np.hstack([phiP, phi_base])

        # jacobian
        Phi_q = asy.get_Phi_q()
        P = asy.p_matrix()                 # rows are p_i^T in the right 4 columns
        Phi_aug = np.zeros((asy.nb + asy.nc, 7*asy.nb))
        Phi_aug[0:asy.nb, 3*asy.nb:] = P   # P only touches p columns in r-p
        Phi_aug[asy.nb:, :] = Phi_q

        # solve and update
        dq = _solveAxb(Phi_aug, -varphi)
        return q_in + dq

    def _project_velocities(v_in, t):
        Phi_q = asy.get_Phi_q()
        P = asy.p_matrix()
        Phi_aug = np.zeros((asy.nb + asy.nc, 7*asy.nb))
        Phi_aug[0:asy.nb, 3*asy.nb:] = P
        Phi_aug[asy.nb:, :] = Phi_q

        # build nu_aug = [0; nu]
        nu = asy.get_nu(t)
        nu_aug = np.zeros(asy.nb + asy.nc)
        nu_aug[asy.nb:] = nu

        rhs = nu_aug - Phi_aug @ v_in
        dv = _solveAxb(Phi_aug, rhs)
        return v_in + dv

    qn = asy.get_q(normalize_euler=True)
    assert np.max(asy.get_phi(0.0)) < 1e-8
    vn = asy.get_qdot() 
    an = np.zeros_like(qn)  # Or some specific IC

    # TODO: Back-calculate qnm1, etc. based on vn initial condition
    qnm1 = deepcopy(qn)
    qnm2 = deepcopy(qn)
    vnm1 = deepcopy(vn)
    vnm2 = deepcopy(vn)

    t = 0.0
    step_num = 0
    q_results = [qn]
    times = [t]

    # x := [an, lambda^P, lambda]
    xn = np.zeros(8*asy.nb + asy.nc)    # TODO: Link this with an
    while t < end_time:
        t = t + dt
        step_num += 1
        print(f"------------------- Time = {t} -------------------")

        # Newton Loop
        # Stage 1a: Compute position and velocity using BDF and most recent acceleration
        ak = an[:]
        xk = xn[:]

        if step_num > 1:
            beta0 = 2/3
            qstar = (4/3)*qn - (1/3)*qnm1
            vstar = (4/3)*vn - (1/3)*vnm1
        else:
            beta0 = 1
            qstar = qn[:]
            vstar = vn[:]

        for k in range(max_inner_its):
            print(f"Step {k}:")

            # Stage 1b: Compute position and velocity predictors using BDF and most recent acceleration
            vk = vstar + ak*beta0*dt
            qk = qstar + vk*beta0*dt

            # Stage 2: Compute residual of NL system
            asy.set_q(qk)
            asy.set_qdot(vk)
            A, b = get_matrix_eq(t)     # Updates q and \dot{q} for all bodies, updates EOM & KCon phi values
            gk = A @ xk - b     # Residual
            
            # Stage 3: Solve linear system to get correction
            a2v = lambda a_test: vstar + a_test*beta0*dt
            v2q = lambda v_test: qstar + v_test*beta0*dt
            jac = get_jacobian(xk, t, a2v, v2q)
            correction = relaxation * _solveAxb(jac, -gk, is_sym=True)

            # Stage 4: Improve quality of approximated solution
            xk = xk + correction
            aprev = ak
            ak = xk[0:asy.nq]
            vkp1 = a2v(ak)      # Velocity vector if solution is accepted
            qkp1 = v2q(vkp1)    # Position vector if solution is accepted
            # Update asyk to confirm constraints are being met
            asy.set_q(qkp1)
            asy.set_qdot(vkp1)

            # Logging
            delta_norm = np.linalg.norm(ak - aprev)
            gk_norm = np.linalg.norm(gk)
            correction_norm = np.linalg.norm(correction)
            phi_norm = np.linalg.norm(asy.get_phi(t))
            print(f"\t|Delta| = {delta_norm}")
            print(f"\t|gk| = {gk_norm}")
            print(f"\t|correction| = {correction_norm}")
            print(f"\t|phi| = {phi_norm}")

            # Early break if converged
            res_norms = np.array([delta_norm, gk_norm, correction_norm, phi_norm])
            if np.all(res_norms < 1e-6):
                print(f"\tTerminating Newton Loop")
                break

            if np.any(np.isnan(res_norms)):
                raise Exception("Overflow")
        
        if np.any(res_norms > error_thres):
            raise Exception(f"Residuals exceed allowable error_thres ({error_thres})")
        
        qnm2 = deepcopy(qnm1)
        qnm1 = deepcopy(qn)
        vnm2 = deepcopy(vnm1)
        vnm1 = deepcopy(vn)

        # One last correction for consistency
        an = ak[:]
        xn = xk[:]
        vn = vstar + an*beta0*dt
        qn = qstar + vn*beta0*dt
        asy.set_q(qn)
        asy.set_qdot(vn)

        # Record
        if step_num % write_increment == 0:
            q_results.append(qn)
            times.append(t)
    
    return q_results, times


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