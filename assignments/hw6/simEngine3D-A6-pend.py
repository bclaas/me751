import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union, List
import matplotlib.pyplot as plt

def tilde(v: np.ndarray):
    """
    Skew-symmetric matrix generator of a vector v
    """
    [x,y,z] = v
    return np.array([[0,-z, y],
                     [z, 0,-x],
                     [-y, x, 0]], float)

def _get_Bmat(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Return the 3x4 matrix B(p,s) = d(A(p) s)/dp for Euler parameters p = [e0,e1,e2,e3].

    p (np.ndarray): Euler parameters [e0, e1, e2, e3].
    s (np.ndarray): A 3-vector in the G-RF frame to be rotated by A(p).

    Returns
    Bmat : (3,4) ndarray
    """
    p = np.asarray(p, dtype=float).reshape(4)
    s = np.asarray(s, dtype=float).reshape(3)
    e0 = p[0]
    e  = p[1:]
    etil = tilde(e)
    e0I = e0 * np.eye(3)

    b1 = (e0I + etil) @ s   # Vector
    b2 = np.outer(e, s) - (e0I + etil) @ tilde(s) # 3x3 matrix
    Bmat = np.concatenate((b1.reshape(-1,1), b2), axis=1) # 3x4 matrix
    return Bmat

def A_to_p(A: np.ndarray):

    # Project A to the nearest element of SO(3) to avoid drift
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:  # enforce right-handedness
        U[:, -1] *= -1
        R = U @ Vt

    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        e0 = 0.25 * S
        e1 = (R[2,1] - R[1,2]) / S
        e2 = (R[0,2] - R[2,0]) / S
        e3 = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            e0 = (R[2,1] - R[1,2]) / S
            e1 = 0.25 * S
            e2 = (R[0,1] + R[1,0]) / S
            e3 = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            e0 = (R[0,2] - R[2,0]) / S
            e1 = (R[0,1] + R[1,0]) / S
            e2 = 0.25 * S
            e3 = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            e0 = (R[1,0] - R[0,1]) / S
            e1 = (R[0,2] + R[2,0]) / S
            e2 = (R[1,2] + R[2,1]) / S
            e3 = 0.25 * S

    p = np.array([e0, e1, e2, e3], dtype=float)
    p /= np.linalg.norm(p)
    if p[0] < 0.0:  # fix overall sign (optional but handy for continuity)
        p = -p
    
    return p

class Orientation:
    def __init__(self, e0, e1, e2, e3):
        """
        e0, e1, e2, e3: Euler parameters
        """
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
    
    @property
    def p(self):
        return np.array([self.e0, self.e1, self.e2, self.e3])
    
    @property
    def A(self):
        [e0, e1, e2, e3] = self.p
        return 2*np.array([
                            [e0**2 + e1**2 - 0.5, e1*e2 - e0*e3, e1*e3 - e0*e2],
                            [e1*e2 + e0*e3, e0**2 + e2**2 - 0.5, e2*e3 - e0*e1],
                            [e1*e3 - e0*e2, e2*e3 + e0*e1, e0**2 + e3**2 - 0.5]
                        ], dtype=float)
    
    @property
    def E(self):
        [e0, e1, e2, e3] = self.p
        return np.array([
            [-e1,  e0,  -e3,  e2],
            [-e2,  e3,   e0, -e1],
            [-e3, -e2,   e1,  e0]
        ], dtype=float)

    @property
    def G(self):
        [e0, e1, e2, e3] = self.p
        return np.array([
            [-e1,  e0,   e3, -e2],
            [-e2, -e3,   e0,  e1],
            [-e3,  e2,  -e1,  e0]
        ], dtype=float)
    
    def set_p(self, p_new: np.ndarray):
        [self.e0, self.e1, self.e2, self.e3] = p_new

@dataclass
class RigidBody:
    name: str
    r: np.ndarray   # (x,y,z) of CG in G-RF
    ori: Orientation
    mass: float = None
    inertia: np.ndarray = None
    _id: int = None
    _is_ground: bool = False

    # TODO: Account for markers outside CG

class KCon:
    def __init__(self, ibody, jbody=None):
        self.ibody = ibody

        if jbody is None:
            # jbody is ground
            # Add dummy part
            [e0, e1, e2, e3] = A_to_p(np.eye(3))
            ground_ori = Orientation(e0, e1, e2, e3)
            ground = RigidBody("Ground", np.zeros(3), ground_ori, _is_ground=True)
            self.jbody = ground
        else:
            self.jbody = jbody
    
    def phi(self, q, t):
        raise NotImplementedError
    
    def phi_r(self):
        raise NotImplementedError
    
    def phi_p(self):
        raise NotImplementedError
    
    def phi_q(self):
        """Return [phi_ri, phi_pi, phi_ri, phi_rj]"""
        [phi_ri, phi_rj] = self.phi_r()
        [phi_pi, phi_pj] = self.phi_p()
        return [phi_ri, phi_pi, phi_rj, phi_pj]
    
    def nu(self, t):
        """
        L1 RHS: \dot{f}(t) 
        (same for all base KCons)
        """
        if isinstance(self.fdot, Callable):
            fdott = self.fdot(t)
        elif isinstance(self.fdot, float):
            fdott = self.fdot
        else:
            print("WARNING: fdot(t) passed to KCon is some unexpected format. Disregarding and setting fdot(t) to 0.0")
            fdott = 0.0

        return fdott

    def gamma(self):
        raise NotImplementedError
    
    
class DP1(KCon):
    def __init__(self,
                 ibody: RigidBody,
                 aibar: np.ndarray,
                 jbody: RigidBody=None,
                 ajbar: np.ndarray=None,
                 f: List[Callable]=None,
                 fdot: List[Callable]=None,
                 fddot: List[Callable]=None):
        """
        ibody (RigidBody): I body
        aibar (np.ndarray): Vector P -> Q in L-RF1
        jbody (RigidBody): J body
        ajbar (np.ndarray): Vector P -> Q in L-RF2
        f (Callable): Dot product offset, as function of time.
        fdot (Callable): Time derivative of f, as function of time.
        fddot (Callable): Second time derivative of f, as function of time.
        """
        super().__init__(ibody, jbody)
        self.aibar = aibar
        self.ajbar = ajbar

        if f is None:
            self.f = lambda t: 0.0
        else:
            self.f = f
        
        if fdot is None:
            self.fdot = lambda t: 0.0
        else:
            self.fdot = fdot
        
        if fddot is None:
            self.fddot = lambda t: 0.0
        else:
            self.fddot = fddot
    
    def phi(self, t):
        """
        ACE: \Phi^{DP1} = \bar{a_i^T} @ A_i^T @ A_j @ \bar{a_j} - f(t) = 0
        """
        phi_ = self.aibar.T @ self.ibody.ori.A.T @ self.jbody.ori.A @ self.ajbar

        if isinstance(self.f, Callable):
            ft = self.f(t)
        elif isinstance(self.f, float):
            ft = self.f
        else:
            print("WARNING: f(t) passed to DP1 is some unexpected format. Disregarding and setting f(t) to 0.0")
            ft = 0.0
        
        return phi_ - ft

    def phi_r(self):
        return [np.zeros(3), np.zeros(3)]
    
    def phi_p(self):
        ai = self.ibody.ori.A @ self.aibar
        aj = self.jbody.ori.A @ self.ajbar

        phi_pi = ai.T @ _get_Bmat(self.ibody.ori.p, self.aibar)
        phi_pj = aj.T @ _get_Bmat(self.jbody.ori.p, self.ajbar)
        return [phi_pi, phi_pj]

    def gamma(self, t, pdoti, pdotj):
        """
        L2 RHS: \ddot{f}(t) - a_j^T \tilde{\omega_i} \tilde{\omega_i} a_i - a_i^T \tilde{\omega_j} \tilde{\omega_j} a_j -2 (\tilde{\omega_i}a_i) \cdot (\tilde{\omega_j}a_j)
                where \tilde{\omega} =: \dot{A}A^T 
        """

        ai = self.ibody.ori.A @ self.aibar
        aj = self.jbody.ori.A @ self.ajbar
        witil = tilde(2*self.ibody.ori.E @ pdoti)   # \tilde{\omega_i} = skew(2*E_i @ \dot{p}_i)
        wjtil = tilde(2*self.jbody.ori.E @ pdotj)   # "   "

        gamma_ = aj.T @ witil @ witil @ ai - ai.T @ wjtil @ wjtil @ aj - 2*(np.dot(witil @ ai, wjtil @ aj))

        if isinstance(self.fddot, Callable):
            fddott = np.array(7*[self.fddot(t)])
        elif isinstance(self.fddot, List):
            fddott = np.array([fii(t) if isinstance(fii, Callable) else 0.0 for fii in self.fddot])
        else:
            print("WARNING: fddot(t) passed to DP1 is some unexpected format. Disregarding and setting fddot(t) to 0.0")
            fddott = np.zeros(7)

        return fddott - gamma_

class DP2(KCon):
    """Constrains distance projection between a vector on one body (aibar) and a point on another"""
    def __init__(self,
                 ibody: RigidBody,
                 aibar: np.ndarray,
                 siPbar: np.ndarray,
                 jbody: RigidBody,
                 sjQbar: np.ndarray,
                 f: List[Callable]=None,
                 fdot: List[Callable]=None,
                 fddot: List[Callable]=None):
        """
        ibody (RigidBody): I body
        aibar (np.ndarray): Vector P -> Q in L-RFi
        siPbar (np.ndarray): Location of Point P on ibody in L-RFi
        jbody (RigidBody): J body
        sjQbar (np.ndarray): Location of Point Q on jbody in L-RFj
        f (Callable): Dot product offset, as function of time.
        fdot (Callable): Time derivative of f, as function of time.
        fddot (Callable): Second time derivative of f, as function of time.
        """
        super().__init__(ibody, jbody)
        self.aibar = aibar
        self.siPbar = siPbar
        self.sjQbar = sjQbar

        if f is None:
            self.f = lambda t: 0.0
        else:
            self.f = f
        
        if fdot is None:
            self.fdot = lambda t: 0.0
        else:
            self.fdot = fdot
        
        if fddot is None:
            self.fddot = lambda t: 0.0
        else:
            self.fddot = fddot
    
    def phi(self, t):
        """
        ACE: \Phi^{DP2} = \bar{a_i^T} @ A_i^T @ d_{ij} - f(t) = 0
                          where d_{ij} = r_j + A_j @ \bar{s_j^Q} - r_i - A_i @ \bar{s_i^P}
        """
        dij = self.jbody.r + self.jbody.ori.A @ self.sjQbar - self.ibody.r - self.ibody.ori.A @ self.siPbar
        phi_ = self.aibar.T @ self.ibody.ori.A.T @ dij

        if isinstance(self.f, Callable):
            ft = self.f(t)
        elif isinstance(self.f, float):
            ft = self.f
        else:
            print("WARNING: f(t) passed to DP2 is some unexpected format. Disregarding and setting f(t) to 0.0")
            ft = 0.0
        
        return phi_ - ft
    
    def phi_r(self):
        phi_ri = -self.aibar.T
        dij = self.jbody.r + self.jbody.ori.A @ self.sjQbar - self.ibody.r - self.ibody.ori.A @ self.siPbar
        aiT = ( self.ibody.ori.A @ self.aibar ).T
        phi_ri = -aiT
        phi_rj = dij.T @ _get_Bmat(self.ibody.ori.p, self.aibar) - aiT @ _get_Bmat(self.ibody.ori.p, self.siPbar)
        return [phi_ri, phi_rj]
    
    def phi_p(self):
        aiT = ( self.ibody.ori.A @ self.aibar ).T
        phi_pi = aiT
        phi_pj = aiT @ _get_Bmat(self.jbody.ori.p, self.sjQbar)
        return [phi_pi, phi_pj]

    def gamma(self, t, rdoti, rdotj, pdoti, pdotj):
        """
        L2 RHS: \ddot{f}(t) - (\tilde{\omega_i} \tilde{\omega_i} a_i) \cdot d - 2 (\tilde{\omega_i} a_i) \cdot (\dot{r_j} - \dot{r_i} + \tilde{\omega_j} s_j - \tilde{\omega_i} s_i) - a_i^T (\tilde{\omega_j}\tilde{\omega_j}s_j \tilde{\omega_i}\tilde{\omega_i}s_i)
                where d =: (r_j + s_j) - (r_i + s_i)
        """

        ai = self.ibody.ori.A @ self.aibar
        si = self.ibody.ori.A @ self.siPbar
        sj = self.jbody.ori.A @ self.sjQbar
        d = self.jbody.r + sj - self.ibody.r - si
        witil = tilde(2*self.ibody.ori.E @ pdoti)   # \tilde{\omega_i} = skew(2*E_i @ \dot{p}_i)
        wjtil = tilde(2*self.jbody.ori.E @ pdotj)   # "   "

        h1 = wjtil @ wjtil @ sj - witil @ witil @ si    # Intermediate value
        gamma_ = np.dot(witil @ witil @ ai, d) - 2*np.dot(witil @ ai, rdotj - rdoti + wjtil @ sj - witil @ si) - ai.T @ h1

        if isinstance(self.fddot, Callable):
            fddott = np.array(7*[self.fddot(t)])
        elif isinstance(self.fddot, List):
            fddott = np.array([fii(t) if isinstance(fii, Callable) else 0.0 for fii in self.fddot])
        else:
            print("WARNING: fddot(t) passed to DP2 is some unexpected format. Disregarding and setting fddot(t) to 0.0")
            fddott = np.zeros(7)

        return fddott - gamma_


class D(KCon):
    """Fixes distance between two points"""
    def __init__(self,
                 ibody: RigidBody,
                 siPbar: np.ndarray,
                 jbody: RigidBody,
                 sjQbar: np.ndarray,
                 f: List[Callable]=None,
                 fdot: List[Callable]=None,
                 fddot: List[Callable]=None):
        """
        ibody (RigidBody): I body
        siPbar (np.ndarray): Location of Point P on ibody in L-RFi
        jbody (RigidBody): J body
        sjQbar (np.ndarray): Location of Point Q on jbody in L-RFj
        f (Callable): Dot product offset, as function of time.
        fdot (Callable): Time derivative of f, as function of time.
        fddot (Callable): Second time derivative of f, as function of time.
        """
        super().__init__(ibody, jbody)
        self.siPbar = siPbar
        self.sjQbar = sjQbar

        if f is None:
            self.f = lambda t: 0.0
        else:
            self.f = f
        
        if fdot is None:
            self.fdot = lambda t: 0.0
        else:
            self.fdot = fdot
        
        if fddot is None:
            self.fddot = lambda t: 0.0
        else:
            self.fddot = fddot
    
    def phi(self, t):
        """
        ACE: \Phi^{D} = d_{ij}^T @ d_{ij} = 0
                          where d_{ij} = r_j + A_j @ \bar{s_j^Q} - r_i - A_i @ \bar{s_i^P}
        """
        dij = self.jbody.r + self.jbody.ori.A @ self.sjQbar - self.ibody.r - self.ibody.ori.A @ self.siPbar
        phi_ = dij.T @ dij

        if isinstance(self.f, Callable):
            ft = self.f(t)
        elif isinstance(self.f, float):
            ft = self.f
        else:
            print("WARNING: f(t) passed to D is some unexpected format. Disregarding and setting f(t) to 0.0")
            ft = 0.0
        
        return phi_ - ft
    
    def phi_r(self):
        dij = self.jbody.r + self.jbody.ori.A @ self.sjQbar - self.ibody.r - self.ibody.ori.A @ self.siPbar
        phi_rj = 2 * dij.T
        phi_ri = -phi_rj
        return [phi_ri, phi_rj]

    def phi_p(self):
        dij = self.jbody.r + self.jbody.ori.A @ self.sjQbar - self.ibody.r - self.ibody.ori.A @ self.siPbar
        phi_pi = -2*dij.T @ _get_Bmat(self.ibody.ori.p, self.siPbar)
        phi_pj = 2*dij.T @ _get_Bmat(self.jbody.ori.p, self.sjQbar)
        return [phi_pi, phi_pj]

    def gamma(self, t, rdoti, rdotj, pdoti, pdotj):
        """
        L2 RHS: \ddot{f}(t) - 2||\dot{d_{ij}}||^2 - 2*d_{ij}^T(\tilde{\omega_j}\tilde{\omega_j}s_j - \tilde{\omega_i}\tilde{\omega_i}s_i)
                where \dot{d_{ij}} = \dot{r_j} - \dot{r_i} + \tilde{\omega_j}s_j - \tilde{\omega_i}s_i 
        """

        si = self.ibody.ori.A @ self.siPbar
        sj = self.jbody.ori.A @ self.sjQbar
        witil = tilde(2*self.ibody.ori.E @ pdoti)   # \tilde{\omega_i} = skew(2*E_i @ \dot{p}_i)
        wjtil = tilde(2*self.jbody.ori.E @ pdotj)   # "   "
        dij = self.jbody.r + sj - self.ibody.r - si
        dijdot = rdotj - rdoti + wjtil @ sj - witil @ si

        gamma_ = dijdot.T @ dijdot - 2*dij.T @ (wjtil @ wjtil @ sj - witil @ witil @ si)
        if isinstance(self.fddot, Callable):
            fddott = np.array(7*[self.fddot(t)])
        elif isinstance(self.fddot, List):
            fddott = np.array([fii(t) if isinstance(fii, Callable) else 0.0 for fii in self.fddot])
        else:
            print("WARNING: fddot(t) passed to D is some unexpected format. Disregarding and setting fddot(t) to 0.0")
            fddott = np.zeros(7)

        return fddott - gamma_


class CD(KCon):
    """Coordinate Difference. Directly constrains a coordinate difference between two points"""
    def __init__(self,
                 c: np.ndarray,
                 ibody: RigidBody,
                 siPbar: np.ndarray,
                 jbody: RigidBody=None,
                 sjQbar: np.ndarray=np.zeros(3),
                 f: List[Callable]=None,
                 fdot: List[Callable]=None,
                 fddot: List[Callable]=None):
        """
        ibody (RigidBody): I body
        c (np.ndarray): Vector (in G-RF) along which to apply constraint 
        siPbar (np.ndarray): Location of Point P on ibody in L-RFi
        jbody (RigidBody): J body
        sjQbar (np.ndarray): Location of Point Q on jbody in L-RFj
        f (Callable): Dot product offset, as function of time.
        fdot (Callable): Time derivative of f, as function of time.
        fddot (Callable): Second time derivative of f, as function of time.
        """
        super().__init__(ibody, jbody)
        self.c = c
        self.siPbar = siPbar
        self.sjQbar = sjQbar

        if f is None:
            self.f = lambda t: 0.0
        else:
            self.f = f
        
        if fdot is None:
            self.fdot = lambda t: 0.0
        else:
            self.fdot = fdot
        
        if fddot is None:
            self.fddot = lambda t: 0.0
        else:
            self.fddot = fddot
    
    def phi(self, t):
        """
        ACE: \Phi^{CD} = c^T @ d_{ij} - f(t) = 0
                          where d_{ij} = r_j + A_j @ \bar{s_j^Q} - r_i - A_i @ \bar{s_i^P}
        """
        dij = self.jbody.r + self.jbody.ori.A @ self.sjQbar - self.ibody.r - self.ibody.ori.A @ self.siPbar
        phi_ = self.c.T @ dij

        if isinstance(self.f, Callable):
            ft = self.f(t)
        elif isinstance(self.f, float):
            ft = self.f
        else:
            print("WARNING: f(t) passed to CD is some unexpected format. Disregarding and setting f(t) to 0.0")
            ft = 0.0
        
        return phi_ - ft

    def phi_r(self):
        phi_rj = self.c.T
        phi_ri = -phi_rj
        return [phi_ri, phi_rj]

    def phi_p(self):
        phi_pi = -self.c.T @ _get_Bmat(self.ibody.ori.p, self.siPbar)
        phi_pj = self.c.T @ _get_Bmat(self.jbody.ori.p, self.sjQbar)
        return [phi_pi, phi_pj]

    def gamma(self, t, pdoti, pdotj):
        """
        L2 RHS: \ddot{f}(t) - c^T - (\tilde{\omega_j}\tilde{\omega_j}s_j - \tilde{\omega_i}\tilde{\omega_i}s_i)
                where \dot{d_{ij}} = \dot{r_j} - \dot{r_i} + \tilde{\omega_j}s_j - \tilde{\omega_i}s_i 
        """

        si = self.ibody.ori.A @ self.siPbar
        sj = self.jbody.ori.A @ self.sjQbar
        witil = tilde(2*self.ibody.ori.E @ pdoti)   # \tilde{\omega_i} = skew(2*E_i @ \dot{p}_i)
        wjtil = tilde(2*self.jbody.ori.E @ pdotj)   # "   "

        gamma_ = self.c.T @ (wjtil @ wjtil @ sj - witil @ witil @ si)
        if isinstance(self.fddot, Callable):
            fddott = np.array(7*[self.fddot(t)])
        elif isinstance(self.fddot, List):
            fddott = np.array([fii(t) if isinstance(fii, Callable) else 0.0 for fii in self.fddot])
        else:
            print("WARNING: fddot(t) passed to CD is some unexpected format. Disregarding and setting fddot(t) to 0.0")
            fddott = np.zeros(7)

        return fddott - gamma_
    


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
            #for kc in j.joints: # Ignore for now. Each joint has one KCon at this juncture.
            rows.append(np.atleast_1d(j.phi(t)))

        return np.concatenate(rows).reshape(-1,)

    def get_Phi_q(self, t: float) -> np.ndarray:
        #m = self.nb + len(self.joints) # TODO: Update for higher-order constraints
        # For now, 1 ACE per KCon because only using primitives. Second half of above needs to change when >1 ACE per KCon
        
        m = len(self.joints)
        Phi_q = np.zeros((m, self.nq))
        row = 0
        for J in self.joints:
            #for kc in j.joints: # Ignore for now. Each joint has one KCon at this juncture.
                # each KCon returns [phi_ri, phi_rj, phi_pi, phi_pj]
            pri, prj = J.phi_r()
            ppi, ppj = J.phi_p()
            # place into global Jacobian
            ii = 7*J.ibody._id
            Phi_q[row, ii:ii+3] = pri
            Phi_q[row, ii+3:ii+7] = ppi

            if not J.jbody._is_ground:
                jj = 7*J.jbody._id
                Phi_q[row, jj:jj+3] = prj
                Phi_q[row, jj+3:jj+7] = ppj

            row += 1

        return Phi_q

    def get_nu(self, t: float) -> np.ndarray:
        rows = []
        for J in self.joints:
            for kc in J.joints:
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
            for kc in J.joints:
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
            phi_base = asy.get_Phi(t)
            jac_base = asy.get_Phi_q(t)    # Jacobian == Phi_q matrix

            m0 = len(phi_base)
            m = m0 + asy.nb

            phi = np.zeros(m)
            phi[0:m0] = phi_base
            jac = np.zeros((m, m))
            jac[0:m0,0:m0+1] = jac_base

            # Add holonomic Euler Parameter normalization constraints
            for ii, bdy in enumerate(asy.bodies):
                p_bdy = bdy.ori.p
                phi[m0+ii-1] = 0.5*(p_bdy.T @ p_bdy - 1)

                row = np.zeros(m)
                row[7*bdy._id+3:7*bdy._id+7] = p_bdy
                jac[m0+ii-1,:] = row
            
            # Dims of jac should be mxm at this point
            # Newton-Raphson
            #correction = np.linalg.solve(jac, phi)
            correction, *_ = np.linalg.lstsq(jac, phi, rcond=None)  # HACK
            q = q - correction

            for idx in range(asy.nb):
                new_r = q[7*idx:7*idx+3]
                new_p = q[7*bdy._id+3:7*bdy._id+7]
                asy.bodies[idx].r = new_r
                asy.bodies[idx].ori.set_p(new_p)

        results.append(np.append(q, t))

        t = t + dt
    
    return np.array(results)

if __name__ == "__main__":
    L = 2
    pen_angle = lambda t: (np.pi/4)*np.cos(2*t)
    theta0 = pen_angle(0)
    rpen = np.array([0, L*np.sin(theta0), -L*np.cos(theta0)])   # in G-RF

    # Rotation matrix between G-RF and L-RF
    f = rpen / np.linalg.norm(rpen)
    h = np.array([1, 0, 0])
    g = np.cross(f, h); g = g / np.linalg.norm(g)
    A = np.zeros((3,3))
    A[:,0] = f; A[:,1] = g; A[:,2] = h

    # Test Q
    Qbar = np.array([-L,0,0])   # Point Q coordinates in L-RF
    Q = A @ Qbar + rpen
    print(Q)

    [e0, e1, e2, e3] = A_to_p(A)
    ori = Orientation(e0, e1, e2, e3)
    pendulum = RigidBody("Pendulum", rpen, ori)
    asy = Assembly()
    asy.add_body(pendulum)
    xcon = CD(np.array([1,0,0]), pendulum, Qbar)
    ycon = CD(np.array([0,1,0]), pendulum, Qbar) 
    zcon = CD(np.array([0,0,1]), pendulum, Qbar)
    DP1a = DP1(ibody=pendulum, aibar=np.array([1,0,0]), ajbar=np.array([1,0,0]), f=np.pi/2) # Keeps pendulum in plane of page
    DP1b = DP1(ibody=pendulum, aibar=np.array([0,0,1]), ajbar=np.array([1,0,0]), f=0.0)     # Keeps pendulum from rotation around its own axis
    DP1mot = DP1(ibody=pendulum, aibar=np.array([1,0,0]), ajbar=np.array([0,0,-1]), f=pen_angle)  # Enforces motion
    asy.add_joint(xcon)
    asy.add_joint(ycon)
    asy.add_joint(zcon)
    asy.add_joint(DP1a)
    asy.add_joint(DP1b)
    asy.add_joint(DP1mot)

    # Run Kinematics
    end_time = 10
    dt = 0.001
    pos_results = run_positionAnalysis(asy, dt=dt, end_time=end_time)

    fig = plt.figure()
    time = pos_results[:,7]
    plt.plot(time, pos_results[:,0], label="Dim-0")
    plt.plot(time, pos_results[:,1], label="Dim-1")
    plt.plot(time, pos_results[:,2], label="Dim-2")
    plt.grid()
    plt.legend()
    plt.show()
