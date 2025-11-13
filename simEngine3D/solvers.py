import time
import numpy as np
from copy import deepcopy
from scipy.linalg import solve, LinAlgError
from .Assembly import Assembly
from .Bodies import RigidBody
from .KCons import KCon, DP1, DP2, D, CD

def torque_from_lam(asy, lam_joint):
    Phiq_j = asy.get_Phi_q(0.0)
    Qc = Phiq_j.T @ lam_joint
    Qp = Qc[3:7]
    tau = 0.5*asy.bodies[0].ori.E @ Qp
    return tau

def _normalize_euler(q):
    nb = len(q) // 7
    qout = np.zeros_like(q)
    qout[0:3*nb] = q[0:3*nb]
    for ii in range(nb):
        a = 3*nb + 4*ii
        p = q[a:a+4]
        pout = p / np.linalg.norm(p)
        qout[a:a+4] = pout
        if np.linalg.norm(pout)-1 > 1e-3:
            raise Exception(f"Norms not norming: {np.linalg.norm(pout)}")
    
    return qout

def _project_positions(asy, max_it=2):
    for _ in range(max_it):
        # residual
        phi_base = asy.get_phi(0.0)        # f(t)=0 for your test
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
        dq, *_ = np.linalg.lstsq(Phi_aug, -varphi, rcond=None)
        q = asy.pack_q() + dq
        # cheap renorm per-body
        for i,b in enumerate(asy.bodies):
            b.r = q[7*i:7*i+3]
            b.ori.set_p(q[7*i+3:7*i+7])

def _project_velocities(asy, v):
    Phi_q = asy.get_Phi_q()
    P = asy.p_matrix()
    Phi_aug = np.zeros((asy.nb + asy.nc, 7*asy.nb))
    Phi_aug[0:asy.nb, 3*asy.nb:] = P
    Phi_aug[asy.nb:, :] = Phi_q

    # build nu_aug = [0; nu]
    nu = asy.get_nu(0.0)
    nu_aug = np.zeros(asy.nb + asy.nc)
    nu_aug[asy.nb:] = nu

    rhs = nu_aug - Phi_aug @ v
    dv, *_ = np.linalg.lstsq(Phi_aug, rhs, rcond=None)
    return v + dv

def run_dynamics(asy, dt, end_time, write_increment=1, max_inner_its=10):

    def update_matrix_eq(asy, t, q, qdot):
        # Make changes to body variables themselves (they feed into A)
        rs = q[0:3*asy.nb]
        ps = q[3*asy.nb:]            
        rdots = qdot[0:3*asy.nb]
        pdots = qdot[3*asy.nb:]
        for bdy in asy.bodies:
            bdy.r = rs[3*bdy._id:3*bdy._id+3]
            bdy.ori.set_p(ps[4*bdy._id:4*bdy._id+4])
            bdy._rdot = rdots[3*bdy._id:3*bdy._id+3]
            bdy._pdot = pdots[4*bdy._id:4*bdy._id+4]

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
            b[asy.nq + bdy._id] = -(bdy._pdot.T @ bdy._pdot)

        # Add gamma_hat to b
        for ii, jnt in enumerate(asy.joints):
            # TODO: Fix this if >1 ACE added per joint
            b[asy.nq + asy.nb + ii] = jnt.gamma(t)
        
        return A, b

    def solveAxb(A,b):
        try:
            x = np.linalg.solve(A, b) # scipy.linalg.solve - Leverages symmetry of A better than np.linalg.solve.
        except:
            x, *_ = np.linalg.lstsq(A, b)
        
        return x

    q_results = []
    times = []

    qn = asy.pack_q()
    vn = np.zeros_like(qn)  # Or some specified IC  
    an = np.zeros_like(qn)  # Or some specific IC

    qnm1 = deepcopy(qn)
    qnm2 = deepcopy(qn)
    vnm1 = deepcopy(vn)
    vnm2 = deepcopy(vn)

    t = 0.0
    xn = np.zeros(8*asy.nb + asy.nc)    # TODO: Link this with an

    while t < end_time:
        t = t + dt
        print(f"------------------- Time = {t} -------------------")

        # Newton Loop
        ak = an
        xk = xn
        for k in range(max_inner_its):
            # Stage 1: Compute position and velocity using BDF and most recent acceleration
            #if k > 1:
            #    beta0 = 2/3
            #    qstar = 1.33333*qnm1 - 0.33333*qnm2
            #    vstar = 1.33333*vnm1 - 0.33333*vnm2
            #else:
            beta0 = 1
            qstar = qnm1
            vstar = vnm1

            qk = qstar + ak*((beta0*dt)**2)
            vk = vstar + ak*beta0*dt

            # Stage 2: Compute residual of NL system
            A, b = update_matrix_eq(asy, t, qk, vk)     # Updates q and \dot{q} for all bodies, updates EOM & KCon phi values
            gk = A @ xk - b     # Residual
            
            # Stage 3: Solve linear system to get correction
            correction = solveAxb(A, -gk)

            # Stage 4: Improve quality of approximated solution
            xk = xk + correction
            aprev = ak
            ak = xk[0:asy.nq]

            # Logging
            delta_norm = np.linalg.norm(ak - aprev)
            gk_norm = np.linalg.norm(gk)
            correction_norm = np.linalg.norm(correction)
            print(f"Step {k}:")
            print(f"\t|Delta| = {delta_norm}")
            print(f"\t|gk| = {gk_norm}")
            print(f"\t|correction| = {correction_norm}")

            # Early break if converged
            res_norms = np.array([delta_norm, gk_norm, correction_norm])
            if np.all(res_norms < 1e-6):
                print(f"\tTerminating Newton Loop")
                break

            if np.isnan(delta_norm) or np.isnan(gk_norm) or np.isnan(correction_norm):
                raise Exception("Overflow")
        
        qnm2 = deepcopy(qnm1)
        qnm1 = deepcopy(qn)
        vnm2 = deepcopy(vnm1)
        vnm1 = deepcopy(vn)

        # One last correction for consistency
        an = ak
        vn = vstar + an*beta0*dt
        qn = qstar + an*((beta0*dt)**2)
        _, _ = update_matrix_eq(asy, t, qn, vn)

        # Record
        q_results.append(qn)
        times.append(t)
    
    return q_results, times


def run_dynamics_hht(asy, dt, end_time, write_increment=1):
    qn = asy.pack_q()
    vn = np.zeros_like(qn)
    qnm1 = qn.copy()
    vnm1 = vn.copy()
    t = 0.0

    # HHT parameter (alpha<=0). alpha=0 -> your current BDF2-like scheme.
    a_hht = -0.10

    # BDF2 "gamma" factor used in your update; 1.0 on the first step, then 2/3.
    alpha_bdf = 1.0

    # KKT sizes
    problem_size = asy.nq + asy.nb + asy.nc
    A = np.zeros((problem_size, problem_size))
    b = np.zeros(problem_size)

    mstart = 0
    mend = 3*asy.nb
    jpstart = mend
    jpend = jpstart + 4*asy.nb

    q_results = []
    times = []

    step_num = 0
    while t < end_time - 1e-15:
        t_next = t + dt

        # BDF2 predictors
        vstar = 1.33333*vn - 0.33333*vnm1
        qstar = 1.33333*qn - 0.33333*qnm1

        # Iterates
        qk = qstar.copy()
        vk = vstar.copy()

        MAX_INNER_ITS = 10
        for inner_it in range(MAX_INNER_ITS):
            qk = _normalize_euler(qk)

            # Matrices at n+1 (qk)
            asy.unpack_q(qk)
            M  = asy.mass_matrix()
            Jp = asy.inertia_matrix()

            # HHT alpha-shifted state for forces/constraints: x_{n+α} = (1+α)x_{n+1} - α x_n
            q_alpha = (1.0 + a_hht)*qk - a_hht*qn
            v_alpha = (1.0 + a_hht)*vk - a_hht*vn
            t_alpha = t_next + a_hht*dt

            # Evaluate Phi, P, Q, gammas at n+α
            asy.unpack_q(q_alpha)
            Phi_q = asy.get_Phi_q()
            Pmat = asy.p_matrix()

            # Assemble KKT
            A.fill(0.0); b.fill(0.0)
            A[mstart:mend, mstart:mend]       = M
            A[jpstart:jpend, jpstart:jpend]   = Jp

            Phi_aug = np.zeros((asy.nb + asy.nc, asy.nq))
            Phi_aug[0:asy.nb, 3*asy.nb:] = Pmat       # Euler-parameter normalization rows
            Phi_aug[asy.nb:, :]          = Phi_q      # Joint rows

            A[jpend:, :jpend] = Phi_aug
            A[:jpend, jpend:] = Phi_aug.T

            # RHS: generalized forces at n+α
            b[:asy.nq] = asy.generalized_forces(t_alpha)

            # gamma^P at n+alpha using v_alpha
            for k in range(asy.nb):
                pdot_k = v_alpha[7*k+3:7*k+7]
                b[asy.nq + k] = -(pdot_k @ pdot_k)

            # gamma_hat for joints at n+α using v_alpha
            if asy.nc > 0:
                b[asy.nq + asy.nb:] = asy.gamma(t_alpha, v_alpha)

            # Solve for qddot and multipliers
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x, *_ = np.linalg.lstsq(A, b, rcond=None)

            a_new = x[:asy.nq]

            # BDF2 corrector (your pattern)
            vk = vstar + alpha_bdf*dt*a_new
            qk = qstar + alpha_bdf*dt*vk

            # Convergence check at n+alpha
            asy.unpack_q((1.0 + a_hht)*qk - a_hht*qn)
            phi = asy.get_phi(t_alpha)
            nu  = asy.get_nu(t_alpha)
            if np.linalg.norm(phi) < 1e-6 and np.linalg.norm(Phi_q @ vk - nu) < 1e-6:
                break
            if inner_it == MAX_INNER_ITS - 1:
                print(f"t={t_next}: inner iterations did not converge.")

        # Accept step
        qnm1, vnm1 = qn, vn
        qn,   vn   = qk.copy(), vk.copy()
        alpha_bdf  = 0.66667  # switch to BDF2 after the first step

        times.append(t)
        q_results.append(qn)

        t = t_next

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