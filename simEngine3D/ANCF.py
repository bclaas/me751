import numpy as np
from numpy.polynomial.legendre import leggauss
from dataclasses import dataclass
from typing import List, Tuple
from .Material import Material


@dataclass
class Node:
    x: float
    y: float
    z: float
    nid: float  # Node ID
    x_fixed = False
    y_fixed = False
    z_fixed = False


class Element:
    """General Element to inherit from"""

    def __init__(self, nodes: List[Node], mat: Material, n_gauss_pts: List[int]=[4, 2, 2]):
        self.nodes = tuple(nodes)
        self.mat = mat
        self.n_gauss_pts = n_gauss_pts
    
    @property
    def volume(self):
        return self.length * self.width * self.height
    
    @property
    def mass(self):
        return self.volume * self.mat.density

    def eval_shape_funcs(self, u, v, w):
        return np.array([s(u,v,w) for s in self.shape_funcs])


class B3_24(Element):
    """B3-24 beam. 3D, 2-node, 4 nodal unknowns per node."""
    def __init__(self, nodes: List[Node], mat: Material, length: float, width: float, height: float, n_gauss_pts: List[int]=[4, 2, 2]):
        # Assumes nodes == [p1, p2]
        assert len(nodes) == 2
        self.n_nodal_unknowns = 8
        super().__init__(nodes, mat, n_gauss_pts)
        self.length = length        # TODO: Eliminate this; calculate based on ICs of nodes.
        self.width = width
        self.height = height
        self.n_gauss_pts = n_gauss_pts

        L = length
        self.shape_funcs = [
            lambda u,v,w: 0.5 - 1.5*(u/L) + 2.0*(u/L)**3,
            lambda u,v,w: 0.125*L - 0.25*u - 0.5*(u**2)/L + (u**3)/(L**2),
            lambda u,v,w: 0.5*v - (u*v)/L,
            lambda u,v,w: 0.5*w - (u*w)/L,
            lambda u,v,w: 0.5 + 1.5*(u/L) - 2.0*(u/L)**3,
            lambda u,v,w: -0.125*L - 0.25*u + 0.5*(u**2)/L + (u**3)/(L**2),
            lambda u,v,w: 0.5*v + (u*v)/L,
            lambda u,v,w: 0.5*w + (u*w)/L,
        ]
        
        self._gauss_pts = []
        # tensor-product Legendre points in physical (u,v,w), with correct weights
        xu, wu = leggauss(n_gauss_pts[0])
        xv, wv = leggauss(n_gauss_pts[1])
        xw, ww = leggauss(n_gauss_pts[2])
        J = self._J()
        for i, xi in enumerate(xu):
            u = 0.5 * self.length * xi
            wi = wu[i]
            for j, xj in enumerate(xv):
                v = 0.5 * self.width * xj
                wj = wv[j]
                for k, xk in enumerate(xw):
                    w = 0.5 * self.height * xk
                    wk = ww[k]
                    wgt = J * wi * wj * wk
                    gp = (u, v, w, wgt)
                    self._gauss_pts.append(gp)

    def basis(self, u, v, w):
        return np.array([1.0, u, v, w, u * v, u * w, u * u, u ** 3], dtype=float)

    def H(self, u, v, w):
        """Jacobian of Shape Function"""
        rtn = np.zeros((8,3))
        rtn[0,0] = -3/(2*self.length) + (6*u**2)/(self.length**3)
        rtn[1,0] = -0.25 - u/self.length + (3*u**2)/(self.length**2)
        rtn[2,0] = -v / self.length
        rtn[2,1] = (0.5*self.length - u) / self.length
        rtn[3,0] = -w / self.length
        rtn[3,2] = (0.5*self.length - u) / self.length
        rtn[4,0] = 3/(2*self.length) - (6*u**2)/(self.length**3)
        rtn[5,0] = -0.25 + u/self.length + (3*u**2)/(self.length**2)
        rtn[6,0] = v / self.length
        rtn[6,1] = (0.5*self.length + u) / self.length
        rtn[7,0] = w / self.length
        rtn[7,2] = (0.5*self.length + u) / self.length
        return rtn
    
    def _J(self):
        return (self.length * self.width * self.height) / 8.0

    def mass_matrix(self):
        mbar = np.zeros((8, 8))
        rho = self.mat.density
        for (u, v, w, wt) in self._gauss_points:
            s = self.eval_shape_funcs(u,v,w)
            mbar += rho * wt * np.outer(s, s)
        M = np.kron(mbar, np.eye(3))   # 24x24 TL mass
        return mbar, M

    def gravity_force(self, gvec):
        rho = self.mat.density
        mg = np.zeros(8)
        for (u, v, w, wt) in self._gauss_points:
            s = self.eval_shape_funcs(u,v,w)
            mg += rho * wt * s
        f = np.zeros(24)
        for a in range(8):
            f[3*a:3*a+3] = mg[a] * gvec
        return f

    def internal_force(self, e):
        """
        e is either (24,) in node-major [x0,y0,z0, x1, ...] or (3,8) with node columns.
        """
        assert isinstance(e, np.ndarray)

        #TODO: Write for general case. Currently expects SVK.

        # current nodal positions N_nodes (3x8)
        if e.ndim == 1 and e.size == 24:
            N_nodes = e.reshape(8, 3).T
        elif e.shape == (3, 8):
            N_nodes = e
        else:
            raise ValueError("e must be (24,) or (3,8)")

        f3x8 = np.zeros((3, 8))
        for (u, v, w, wt) in self._gauss_points:
            Huvw = self.H(u,v,w)
            su = Huvw[:,0]
            sv = Huvw[:,1]
            sw = Huvw[:,2]
            ru = N_nodes @ su
            rv = N_nodes @ sv
            rw = N_nodes @ sw
            F  = np.column_stack((ru, rv, rw))          # deformation gradient
            P = self.mat.PK1(F)
            for a in range(8):
                f3x8[:, a] += wt * (su[a] * P[:, 0] + sv[a] * P[:, 1] + sw[a] * P[:, 2])

        fout = np.zeros(24)
        for a in range(8):
            fout[3*a:3*a+3] = f3x8[:, a]
        return fout

    def position(self, e, u, v, w):
        if e.ndim == 1 and e.size == 24:
            N_nodes = e.reshape(8, 3).T
        else:
            N_nodes = e
        
        return N_nodes @ self.eval_shape_funcs(u,v,w)
    
    def generalize_external_force(self, f_ext, u, v, w):
        s = self.eval_shape_funcs(u,v,w)
        f = np.zeros(3*self.n_nodal_unknowns)
        for ii in range(self.n_nodal_unknowns):
            f[3*ii:3*ii+3] = s[ii] * f_ext
        return f