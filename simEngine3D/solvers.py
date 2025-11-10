import time
import numpy as np
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

def run_dynamics(asy, dt, end_time, write_increment=1):    
    qn = asy.pack_q()
    vn = np.zeros_like(qn)
    qdotdot = np.zeros_like(qn)
    qnm1 = qn.copy()
    vnm1 = vn.copy()
    t = 0.0
    
    times = []
    q_results = []
    #r_hist = []
    #w_hist = []
    #tau_hist = []
    #vviol_hist = []

    energy0, _, _ = asy.get_energy()

    problem_size = 8*asy.nb + asy.nc
    A = np.zeros((problem_size, problem_size))
    b = np.zeros(problem_size)
    mstart = 0                      # Start index for M matrix within A
    mend = 3*asy.nb                 # Last index for M matrix within A
    jpstart = mend                  # Start index for Jp matrix within A
    jpend = jpstart + 4*asy.nb      # Last index for Jp matrix within A

    step_num = 0
    qstar = qn
    vstar = vn
    alpha = 1.0     # changes to 2/3 after first time step

    solver_start = time.perf_counter()
    while t < end_time - 1e-15:
        t = t + dt

        # Build L2 matrix equation
        # | M_aug    Phi_aug^T  | * | qdotdot | = | f_aug   |
        # | Phi_aug  0          |   | lam     |   | gam_aug |
        # aka Ax = b

        # M_aug
        A[mstart:mend, mstart:mend] = asy.mass_matrix()
        A[jpstart:jpend, jpstart:jpend] = asy.inertia_matrix()

        # Phi_aug (build independently then add Phi_aug & Phi_aug.T to A)
        Phi_aug = np.zeros((asy.nb+asy.nc, 7*asy.nb))
        Phi_aug[0:asy.nb, mend:jpend] = asy.p_matrix()
        Phi_q = asy.get_Phi_q()
        Phi_aug[asy.nb::] = Phi_q
        A[jpend::, 0:jpend] = Phi_aug
        A[0:jpend, jpend::] = Phi_aug.T

        # Update vector b
        # Add generalized forces
        b[0:asy.nq] = asy.generalized_forces(t)

        # Add gamma^P to b
        for bdy in asy.bodies:
            b[asy.nq + bdy._id] = -2 * bdy._pdot.T @ bdy._pdot

        # Add gamma_hat to b
        for ii, jnt in enumerate(asy.joints):
            # TODO: Fix this if >1 ACE added per joint
            b[asy.nq + asy.nb + ii] = jnt.gamma(t)

        try:
            #x = solve(A, b, assume_a="sym", check_finite=False) # scipy.linalg.solve - Leverages symmetry of A better than np.linalg.solve.
            x = np.linalg.solve(A,b)
        except LinAlgError:
            try:
                x, *_ = np.linalg.lstsq(A, b)
            except np.linalg.LinAlgError as err:
                print(f"(t = {t}) Both matrix solvers have failed.")
                raise err
            
        qdotdot = x[0:asy.nq]
        lam = x[asy.nq::]

        vnm1 = vn
        qnm1 = qn
        vn = vstar + alpha*dt*qdotdot
        qn = qstar + ((alpha*dt)**2)*qdotdot

        # Accuracy Checks
        solve_fail = False

        phi = asy.get_phi(t)
        if np.any(phi > 1e-4):
            solve_fail = True
            print("Non-converging constraint equations. Error incoming.")
            print(f"\tphi = {phi}")

        nu = asy.get_nu(t)
        vel_resd = Phi_q @ vn - nu
        # if np.any(vel_resd > 1e-3):
        #     solve_fail = True
        #     print("System state moving into constraint Jacobian. Error incoming.")
        #     print(f"\tPhi_q @ v - nu = {vel_resd}")

        tot, kin, pot = asy.get_energy(vn)
        energy_resd = np.abs(tot - energy0)
        # if energy_resd > 1e-4:
        #     solve_fail = True
        #     print("Energy imbalance detected. Error incoming.")
        #     print(f"\tKinetic = {kin}")
        #     print(f"\tPotential = {pot}")
        #     print(f"\tTotal = {tot}")
        
        if solve_fail:
            raise Exception(f"Solver failed at t = {t} sec.")
        
        # Record results
        if step_num % write_increment == 0:
            times.append(t)
            q_results.append(qn)
            # TODO: Calculate reaction forces if desired

        # Update qstar & vstar. Update system state to reflect qstar and vstar.
        qstar = 1.3333*qn - 0.3333*qnm1
        vstar = 1.3333*vn - 0.3333*vnm1
        asy.unpack_q(qstar)     # Updates r and p of bodies
        rdots = vstar[0:3*asy.nb]
        pdots = vstar[3*asy.nb:]
        for bdy in asy.bodies:
            bdy._rdot = rdots[3*bdy._id:3*bdy._id+3]
            bdy._pdot = pdots[4*bdy._id:4*bdy._id+4]

        alpha = 0.66667     # Only 1.0 for first time step
        step_num += 1

    sim_time = time.perf_counter() - solver_start
    print(f"{sim_time = }")
    #return np.array(ts), np.array(r_hist), np.array(w_hist), np.array(tau_hist), np.array(vviol_hist), sim_time
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