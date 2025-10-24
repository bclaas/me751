import numpy as np
from typing import List, Tuple
from .Bodies import RigidBody
from .KCons import KCon, DP1, DP2, D, CD

class Assembly:
    def __init__(self):
        self.bodies: List[RigidBody] = []
        self.joints: List[KCon] = []    # TODO: Add higher-order joints/constraints
        self.forces: List = []
        self.grav = np.array([0.0, 0.0, 0.0])

    def add_body(self, body: RigidBody):
        body._id = len(self.bodies)
        self.bodies.append(body)

    def add_joint(self, joint: KCon):
        self.joints.append(joint)

    def add_grav(self, g: np.ndarray):
        assert len(g) == 3
        self.grav = g
    
    # TODO: Make `Force` class

    @property
    def nb(self):
        return len(self.bodies)
    
    @property
    def nq(self): return 7 * self.nb

    def pack_q(self) -> np.ndarray:
        q = np.zeros(self.nq)
        for k,b in enumerate(self.bodies):
            i = 7*k
            q[i:i+3] = b.r
            q[i+3:i+7] = b.p
        return q

    def unpack_q(self, q: np.ndarray):
        for k,b in enumerate(self.bodies):
            i = 7*k
            b.r = q[i:i+3].copy()
            b.ori.p = q[i+3:i+7].copy()

    def get_Phi(self, t: float) -> np.ndarray:
        rows = []
        for j in self.joints:
            for kc in j.kcons:
                rows.append(np.atleast_1d(kc.phi(t)))

        return np.concatenate(rows).reshape(-1,)

    def get_Phi_q(self, t: float) -> np.ndarray:
        #m = self.nb + len(self.kcons) # TODO: Update for higher-order constraints
        # For now, 1 ACE per KCon because only using primitives. Second half of above needs to change when >1 ACE per KCon
        
        m = len(self.joints)
        Phi_q = np.zeros((m, self.nq))
        row = 0
        for J in self.joints:
            for kc in J.kcons:
                # each KCon returns [phi_ri, phi_rj, phi_pi, phi_pj]
                pri, prj = kc.phi_r()
                ppi, ppj = kc.phi_p()
                # place into global Jacobian
                ii = 7*kc.ibody._id
                jj = 7*kc.jbody._id
                Phi_q[row, ii:ii+3] = pri
                Phi_q[row, jj:jj+3] = prj
                Phi_q[row, ii+3:ii+7] = ppi
                Phi_q[row, jj+3:jj+7] = ppj
                row += 1
        
        # Enforce quaternion renormalization
        #for pp in range(self.nb):
        #    idx1 = 7*pp+3
        #    idx2 = idx1+4
        #    Phi_q[row, idx1:idx2] = np.ones(4)
        #    row += 1

        return Phi_q

    def get_nu(self, t: float) -> np.ndarray:
        rows = []
        for J in self.joints:
            for kc in J.kcons:
                rows.append(np.atleast_1d(kc.nu(t)))
        return np.concatenate(rows).reshape(-1,)

    def gamma(self, t: float, qdot: np.ndarray) -> np.ndarray:
        # split qdot into per-body (rdot, pdot)
        rdots, pdots = [], []
        for k in range(self.nb):
            i = 7*k
            rdots.append(qdot[i:i+3])
            pdots.append(qdot[i+3:i+7])
        rows = []
        for J in self.joints:
            for kc in J.kcons:
                ib = self._body_index[kc.ibody._id]
                jb = self._body_index[kc.jbody._id]
                rows.append(np.atleast_1d(kc.gamma(
                    t,
                    rdots[ib], rdots[jb],
                    pdots[ib], pdots[jb]
                )))
        return np.concatenate(rows).reshape(-1,)

    def mass_matrix(self) -> np.ndarray:
        """Block-diagonal translational part; rotational mapping filled later."""
        M = np.zeros((self.nq, self.nq))
        for k,b in enumerate(self.bodies):
            ii = 7*k
            M[ii:ii+3, ii:ii+3] = b.mass * np.eye(3)
            # TODO: rotational 4x4 block needs r-p mapping (E/G) per notes; fill later

        return M

    def generalized_forces(self, t: float) -> np.ndarray:
        # TODO: Add generalized forces for p
        Q = np.zeros(self.nq)
        for b in self.bodies:
            Qr = np.zeros(3)
            for f in self.forces:
                Qr += f

            ii = 7*b._id
            Q[ii:ii+3] += Qr
            Q[ii:ii+3] += self.grav * b.mass
            Q[ii+3:ii+7] += np.zeros(0)

        return Q
