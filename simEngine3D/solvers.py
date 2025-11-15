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

def _solveAxb(A, b, is_sym=False):
    try:
        if is_sym:
            x = scipy.linalg.solve(A, b, assume_a="sym", check_finite=False)
        else:
            x = scipy.linalg.solve(A, b, check_finite=False)
    except:
        x, *_ = np.linalg.lstsq(A, b)
    
    return x

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


def run_dynamics(asy, dt, end_time, write_increment=1, max_inner_its=10, relaxation=1.0, error_thres=np.inf):

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

    def get_jacobian(x0, asy, t, a2v, v2q):
        # Perturb the system of equations to get Jacobian
        # Use DEEPCOPY of assembly so state of system outside this function is unaffected
        casy = deepcopy(asy)
        m = len(x0)
        n = 8*casy.nb + casy.nc
        jac = np.empty((m,n), dtype=float)
        eps = np.sqrt(np.finfo(float).eps)
        #eps = 0.1

        def g_perturbed(idx, perturbation):
            x = x0[:]
            x[idx] += perturbation
            a_test = x[0:casy.nq]
            v_test = a2v(a_test)
            q_test = v2q(v_test)
            A, b = update_matrix_eq(casy, t, q_test, v_test)
            return A @ x - b
        
        for idx in range(m):
            h = eps * max(1.0, abs(x0[idx]))
            gu = g_perturbed(idx, h)    # Correction when x_idx perturbed upward
            gd = g_perturbed(idx, -h)    # Correction when x_idx perturbed downward
            jac[:, idx] = (gu - gd)/(2*h)

        return jac


    qn = asy.pack_q()
    vn = np.zeros_like(qn)  # Or some specified IC  
    an = np.zeros_like(qn)  # Or some specific IC

    qnm1 = deepcopy(qn)
    qnm2 = deepcopy(qn)
    vnm1 = deepcopy(vn)
    vnm2 = deepcopy(vn)

    t = 0.0
    step_num = 0
    q_results = [qn]
    times = [t]

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

            # Stage 1b: Compute position and velocity using BDF and most recent acceleration
            vk = vstar + ak*beta0*dt
            qk = qstar + vk*beta0*dt
            
            # Stage 2: Compute residual of NL system
            A, b = update_matrix_eq(asy, t, qk, vk)     # Updates q and \dot{q} for all bodies, updates EOM & KCon phi values
            gk = A @ xk - b     # Residual
            
            # Stage 3: Solve linear system to get correction
            a2v = lambda a_test: vstar + a_test*beta0*dt
            v2q = lambda v_test: qstar + v_test*(beta0*dt)
            jac = get_jacobian(xk, asy, t, a2v, v2q)
            correction = relaxation * _solveAxb(jac, -gk, is_sym=True)

            # Stage 4: Improve quality of approximated solution
            xk = xk + correction
            aprev = ak
            ak = xk[0:asy.nq]

            # Logging
            delta_norm = np.linalg.norm(ak - aprev)
            gk_norm = np.linalg.norm(gk)
            correction_norm = np.linalg.norm(correction)
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
        _, _ = update_matrix_eq(asy, t, qn, vn)

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