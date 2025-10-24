import numpy as np
from typing import Callable, Dict, Optional, Tuple, List
from .Bodies import RigidBody
from .Orientation import Orientation, vec2quat, tilde, A_to_p

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
    