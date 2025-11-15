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
    def nq(self):
        return 7 * self.nb

    @property
    def nc(self):
        return len(self.joints)   # TODO: Adjust when each joint has >1 ACE

    def pack_q(self) -> np.ndarray:
        q = np.zeros(self.nq)
        for bdy in self.bodies:
            r_id = 3*bdy._id                # Starting column of transl. DOFs within r-p form L2 matrix 'A'
            p_id = 4*bdy._id + 3*self.nb    # Starting column of rot. DOFs within r-p form L2 matrix 'A'
            q[r_id:r_id+3] = bdy.r
            p = bdy.ori.p
            q[p_id:p_id+4] = p / np.linalg.norm(p)  # Ensure euler parameter normalization
        return q

    def unpack_q(self, q: np.ndarray):
        for bdy in self.bodies:
            r_id = 3*bdy._id                # Starting column of transl. DOFs within r-p form L2 matrix 'A'
            p_id = 4*bdy._id + 3*self.nb    # Starting column of rot. DOFs within r-p form L2 matrix 'A'
            bdy.r = q[r_id:r_id+3].copy()
            p = q[p_id:p_id+4].copy()
            bdy.ori.set_p(p / np.linalg.norm(p))  # Ensure euler parameter normalization

    def get_phi(self, t: float) -> np.ndarray:
        rows = []
        for J in self.joints:
            # TODO: Account for joints with >1 ACE
            phij = J.phi(t)
            rows.append((phij))

        return np.array(rows)

    def get_Phi_q(self) -> np.ndarray:
        # For now, 1 ACE per KCon because only using primitives. Second half of above needs to change when >1 ACE per KCon
        
        Phi_q = np.zeros((self.nc, self.nq))
        row = 0
        for J in self.joints:
            # TODO: Account for joints with >1 ACE
            # each KCon returns [phi_ri, phi_rj, phi_pi, phi_pj]
            pri, prj = J.phi_r()
            ppi, ppj = J.phi_p()
            # place into global Jacobian
            r_id = 3*J.ibody._id                # Starting column of transl. DOFs within r-p form L2 matrix 'A'
            p_id = 4*J.ibody._id + 3*self.nb    # Starting column of rot. DOFs within r-p form L2 matrix 'A'
            Phi_q[row, r_id:r_id+3] = pri
            Phi_q[row, p_id:p_id+4] = ppi

            if not J.jbody._is_ground:
                r_id = 3*J.jbody._id                # Starting column of transl. DOFs within r-p form L2 matrix 'A'
                p_id = 4*J.jbody._id + 3*self.nb    # Starting column of rot. DOFs within r-p form L2 matrix 'A'
                Phi_q[row, r_id:r_id+3] = prj
                Phi_q[row, p_id:p_id+4] = ppj

            row += 1

        return Phi_q

    def get_nu(self, t: float) -> np.ndarray:
        rows = []
        for J in self.joints:
            # TODO: Account for joints with >1 ACE
            rows.append(np.atleast_1d(J.nu(t)))

        return np.concatenate(rows).reshape(-1,)

    def gamma(self, t: float, qdot: np.ndarray) -> np.ndarray:
        """split qdot into per-body (rdot, pdot)"""
        # Assumes pdot and rdot already set for body
        rows = []
        for kc in self.joints:
            rows.append(np.atleast_1d(kc.gamma(t)))

        return np.concatenate(rows).reshape(-1,)

    def mass_matrix(self) -> np.ndarray:
        """Block-diagonal mass matrix. Mass only, no inertia"""
        M = np.zeros((3*self.nb, 3*self.nb))
        for bdy in self.bodies:
            if bdy._is_ground:
                continue

            i1 = 3*bdy._id
            i2 = i1 + 3
            M[i1:i2, i1:i2] = bdy.mass * np.eye(3)
        
        return M
    
    def inertia_matrix(self) -> np.ndarray:
        """Mass moment of inertia tensor. Often denoted J^p"""
        Jp = np.zeros((4*self.nb, 4*self.nb))
        for bdy in self.bodies:
            if bdy._is_ground:
                continue

            G = bdy.ori.G
            Jpb = 4 * G.T @ bdy.inertia @ G
            i1 = 4*bdy._id
            i2 = i1 + 4
            Jp[i1:i2, i1:i2] = Jpb
        
        return Jp

    def p_matrix(self) -> np.ndarray:
        """
        P (euler parameter) matrix
        | [e0, e1, e2, e3] [0,   0,  0,  0]     |
        | [0,   0,  0,  0] [e0, e1, e2, e3] ... | <- One row per body
        |        ...             ...        ... |
        """
        P = np.zeros((self.nb, 4*self.nb))
        for bdy in self.bodies:
            if bdy._is_ground:
                continue
            
            c1 = 4*bdy._id
            c2 = c1 + 4
            P[bdy._id, c1:c2] = bdy.ori.p

        return P

    def generalized_forces(self, t: float) -> np.ndarray:
        Q = np.zeros(self.nq)
        for bdy in self.bodies:
            if bdy._is_ground:
                continue

            # Forces (Translational)
            r_id = 3*bdy._id                # Starting column of transl. DOFs within r-p form L2 matrix 'A'
            Fb = np.zeros(3)
            # TODO
            # for f in self.forces:
            #     Fb += f
            Fb += self.grav * bdy.mass
            Q[r_id:r_id+3] = Fb

            # Torques (Rotational)
            p_id = 4*bdy._id + 3*self.nb    # Starting column of rot. DOFs within r-p form L2 matrix 'A'
            Tb = np.zeros(4)
            # TODO
            # for torq in self.torques:
            #     Tb += torq
            E = bdy.ori.E

            Tb += 8.0 * (E.T @ (bdy.inertia @ (E @ bdy._pdot)))   # 4x1
            Q[p_id:p_id+4] = Tb

        return Q
    
    def get_energy(self, v=None):
            kin = 0.0
            pot = 0.0
            for k,b in enumerate(self.bodies):
                i = 7*k

                if v is not None:
                    vr = v[i:i+3]
                    vp = v[i+3:i+7]
                    A  = b.ori.A
                    E  = b.ori.E
                    Jg = A @ b.inertia @ A.T
                    omega = 2.0 * (E @ vp)
                    kin += 0.5 * b.mass * (vr @ vr) + 0.5 * (omega @ (Jg @ omega))

                pot += b.mass * (self.grav @ b.r)

            return kin+pot, kin, pot
