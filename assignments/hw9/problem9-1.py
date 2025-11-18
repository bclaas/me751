import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import sys
sys.path.append("C:/ME751/me751")
from simEngine3D.Material import SVK, Material
from simEngine3D.ANCF import B3_24, Node

np.set_printoptions(suppress=True, linewidth=140, precision=8)

from typing import List, Tuple


@dataclass
class Node:
    x: float
    y: float
    z: float
    nid: float  # Node ID


class Element:
    """General Element to inherit from"""

    def __init__(self, nids: List[int], mat: Material):
        self.node_ids = tuple(nids)
        self.mat = mat


class B3_24(Element):
    """B3-24 beam. 3D, 2-node, 4 nodal unknowns per node."""
    def __init__(self, nodes: List[Node], mat: Material, length: float, width: float, height: float):
        # Assumes nodes == [p1, p2]
        assert len(nodes) == 2
        self.n_nodal_unknowns = 8
        super().__init__(nodes, mat)
        self.length = length        # TODO: Eliminate this; calculate based on ICs of nodes.
        self.width = width
        self.height = height
        # TODO: Compute x-sectional areas and moments of inertia?

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
    
    @property
    def volume(self):
        return self.length * self.width * self.height
    
    @property
    def mass(self):
        return self.volume * self.mat.density

    def eval_shape_funcs(self, u, v, w):
        return np.array([s(u,v,w) for s in self.shape_funcs])

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

    def _gauss_points(self, nu=4, nv=2, nw=2):
        # tensor-product Legendre points in physical (u,v,w), with correct weights
        xu, wu = leggauss(nu)
        xv, wv = leggauss(nv)
        xw, ww = leggauss(nw)
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
                    yield u, v, w, (J * wi * wj * wk)

    def mass_matrix(self, nu=4, nv=2, nw=2):
        mbar = np.zeros((8, 8))
        rho = self.mat.density
        for u, v, w, wt in self._gauss_points(nu, nv, nw):
            s = self.eval_shape_funcs(u,v,w)
            mbar += rho * wt * np.outer(s, s)
        M = np.kron(mbar, np.eye(3))   # 24x24 TL mass
        return mbar, M

    def gravity_force(self, gvec, nu=4, nv=2, nw=2):
        rho = self.mat.density
        mg = np.zeros(8)
        for u, v, w, wt in self._gauss_points(nu, nv, nw):
            s = self.eval_shape_funcs(u,v,w)
            mg += rho * wt * s
        f = np.zeros(24)
        for a in range(8):
            f[3*a:3*a+3] = mg[a] * gvec
        return f

    def internal_force(self, e, nu=5, nv=3, nw=3):
        """
        e is either (24,) in node-major [x0,y0,z0, x1, ...] or (3,8) with node columns.
        """
        assert isinstance(e, np.ndarray)
        # material

        #TODO: Write for general case. Currently expects SVK.
        mat = self.mat

        # current nodal positions N_nodes (3x8)
        if e.ndim == 1 and e.size == 24:
            N_nodes = e.reshape(8, 3).T
        elif e.shape == (3, 8):
            N_nodes = e
        else:
            raise ValueError("e must be (24,) or (3,8)")

        f3x8 = np.zeros((3, 8))
        for u, v, w, wt in self._gauss_points(nu, nv, nw):
            Huvw = self.H(u,v,w)
            su = Huvw[:,0]
            sv = Huvw[:,1]
            sw = Huvw[:,2]
            ru = N_nodes @ su
            rv = N_nodes @ sv
            rw = N_nodes @ sw
            F  = np.column_stack((ru, rv, rw))          # deformation gradient
            P = mat.PK1(F)
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
    
@dataclass
class Force:
    fvec: np.ndarray
    u: float
    v: float
    w: float

# Material-model-independent tensors
def cauchy_green_strain(F: np.ndarray):
    # RIGHT Cauchy-Green Strain Tensor
    # Often denoted C
    # a.k.a. deformation tensor
    return F.T @ F

def green_lagrange_strain(F: np.ndarray):
    # Often denoted E
    # Symmetric
    # True strain measure; not affected by rigid body rotation; vanishes if undeformed
    return 0.5*(cauchy_green_strain(F) - np.eye(len(F)))

class Material:
    def __init__(self, density: float):
        self.density = density

class SVK(Material):
    # Saint-Venant-Kirchhoff
    # Materially (i.e. in terms of material properties) linear; geometrically non-linear
    def __init__(self, density, youngs, poissons):
        super().__init__(density)
        self.lmbda = youngs*poissons/((1+poissons)*(1-2*poissons))
        self.mu = youngs / (2*(1+poissons))
    
    def strain_energy_density(self, F):
        E = green_lagrange_strain(F)
        trE = np.trace(E)
        return 0.5*self.lmbda*(trE**2) + self.mu*np.trace(E @ E)
    
    def PK1(self, F):
        # First Piola-Kirchhoff stress tensor. Often denoted P.
        S = self.PK2(F)
        return F @ S
    
    def PK2(self, F):
        # Second Piola-Kirchhoff stress tensor. Often denoted S.
        E = green_lagrange_strain(F)
        return self.lmbda*np.trace(E)*np.eye(3) + 2*self.mu*E
    
def f_internal_free(elem, e_full, mask_free, nu=4, nv=3, nw=3):
    """Internal generalized force restricted to free DOFs."""
    f_int = elem.internal_force(e_full, nu=nu, nv=nv, nw=nw)
    return f_int[mask_free]

# HACK
def numerical_tangent(elem, e_full, mask_free, h=1e-8, nu=5, nv=3, nw=3):
    """Finite-difference tangent K_ff = d(f_int_free)/d(e_free) at the current state."""
    nf = mask_free.sum()
    K = np.zeros((nf, nf))
    f0 = f_internal_free(elem, e_full, mask_free, nu=nu, nv=nv, nw=nw)
    idx_free = np.where(mask_free)[0]
    for j in range(nf):
        ej = idx_free[j]
        e_pert = e_full.copy()
        step = h * max(1.0, abs(e_full[ej]))
        e_pert[ej] += step
        fj = f_internal_free(elem, e_pert, mask_free, nu=nu, nv=nv, nw=nw)
        K[:, j] = (fj - f0) / step
    return K


def backward_euler_step(elem, M_ff, e_n_free, v_n_free, e_full_n, mask_free, dt, g_free, forces, tol=1e-5, maxit=30):
    """
    One Backward Euler step with a quasi-Newton solve on the free DOFs.

    elem : B3_24
        Element providing force evaluations/mappings.
    M_ff : (n_free, n_free) ndarray
        Reduced mass matrix for the free DOFs.
    e_n_free : (n_free,) ndarray
        Free coordinates at time n.
    v_n_free : (n_free,) ndarray
        Free velocities at time n.
    e_full_n : (24,) ndarray
        Full coordinate vector at time n (contains fixed DOFs too).
    mask_free : (24,) boolean ndarray
        Free-DOF mask.
    dt : float
        Time step size (s).
    g_free : (n_free,) ndarray
        Gravity generalized force restricted to free DOFs (constant).
    tol : float, optional
        Nonlinear residual norm tolerance.
    maxit : int, optional
        Maximum Newton iterations.
    """
    nf = M_ff.shape[0]
    y = np.hstack([e_n_free.copy(), v_n_free.copy()])  # initial guess: carry-over
    for it in range(maxit):
        e_np1_free = y[:nf]
        v_np1_free = y[nf:]
        
        # Internal forces
        e_full_np1 = e_full_n.copy()
        e_full_np1[mask_free] = e_np1_free
        fint = f_internal_free(elem, e_full_np1, mask_free)

        # External forces
        fext = g_free[:]
        for frce in forces:
            fexti = elem.generalize_external_force(frce.fvec, u=frce.u, v=frce.v, w=frce.w)
            fext += fexti[mask_free]

        r1 = e_np1_free - e_n_free - dt * v_np1_free
        r2 = M_ff @ (v_np1_free - v_n_free) - dt * (fint + fext)
        r = np.hstack([r1, r2])
        nr = np.linalg.norm(r)
        if nr < tol:
            return e_np1_free, v_np1_free, it, nr
        K_ff = numerical_tangent(elem, e_full_np1, mask_free)

        # Build Jacobian
        J = np.zeros((2*nf, 2*nf))
        J[:nf, :nf] = np.eye(nf)
        J[:nf, nf:] = -dt * np.eye(nf)
        J[nf:, :nf] = -dt * K_ff
        J[nf:, nf:] = M_ff

        try:
            delta = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            # fallback: least-squares
            delta, *_ = np.linalg.lstsq(J, -r, rcond=None)

        y = y + delta
    
    # Return anyway
    return e_np1_free, v_np1_free, it, nr

if __name__ == "__main__":
    # Geometry & material
    L = 0.5
    W = 0.003
    H = 0.003
    rho = 7700.0
    youngs = 2.0e11
    poissons = 0.3
    gvec = np.array([0.0, 0.0, -9.81])

    # Material (SVK)
    mat = SVK(density=rho, youngs=youngs, poissons=poissons)

    # Definbe beam
    n1 = Node(0.0, 0.0, 0.0, nid=1)
    n2 = Node(L, 0.0, 0.0, nid=2)
    elem = B3_24(nodes=[n1, n2], mat=mat, length=L, width=W, height=H)

    # Initial nodal unknowns
    N = np.column_stack([
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]).reshape(3, 8)
    e0 = np.ravel(N)
    v0 = np.zeros_like(e0)

    # Boundary conditions
    mask_free = np.zeros(24, dtype=bool)
    mask_free[12:24] = True

    # Gravity -- free node only
    g_free = elem.gravity_force(gvec); g_free = g_free[mask_free]

    # Mass matrix
    mij, M = elem.mass_matrix(nu=4, nv=2, nw=2)
    M_ff = M[np.ix_(mask_free, mask_free)]

    # Gravity generalized force
    f_grav = elem.gravity_force(gvec, nu=4, nv=2, nw=2)

    # Internal force
    f_int = elem.internal_force(N, nu=4, nv=2, nw=2)

    t0 = 0.0
    tf = 0.5
    dt = 5e-4
    nt = int(np.round((tf - t0)/dt))
    times = np.linspace(t0+dt, tf, nt)

    e_hist = np.zeros((nt+1, 24))
    v_hist = np.zeros((nt+1, 24))
    e_hist[0] = e0
    v_hist[0] = v0

    e_n = e0.copy()
    v_n = v0.copy()

    tip_xyz = np.zeros((nt+1, 3))
    tip_xyz[0] = elem.position(e0.reshape(3,8), +L/2.0, 0.0, 0.0)

    start_time = time.perf_counter()
    for k, t_np1 in enumerate(times, start=1):
        e_n_free = e_n[mask_free]
        v_n_free = v_n[mask_free]

        tip_force = Force(np.array([0, 0, -1 + np.cos(20*np.pi*t_np1)]), 1, 0, 0)
        forces = [tip_force]
        
        e_np1_free, v_np1_free, it, nr = backward_euler_step(elem, M_ff, e_n_free, v_n_free, e_n.copy(), mask_free, dt, g_free, forces=forces)
        e_np1 = e_n.copy()
        v_np1 = v_n.copy()
        e_np1[mask_free] = e_np1_free
        v_np1[mask_free] = v_np1_free

        e_hist[k] = e_np1
        v_hist[k] = v_np1
        e_n = e_np1
        v_n = v_np1

        tip_xyz[k] = elem.position(e_np1.reshape(3,8), +L/2.0, 0.0, 0.0)

        if (k % 100) == 0:
            print(f"t={t_np1:.6f}  Newton it={it}  ||r||={nr:.2e}  tip z={tip_xyz[k,2]:.6e}")
    
    runtime = time.perf_counter() - start_time
    print(f"Solver Time = {runtime}")
    
    z_pos_tip = [tip_xyz[k][-1] for k in range(tip_xyz)]
    fig = plt.figure()
    plt.plot(times, z_pos_tip)
    plt.grid()
    plt.xlabel("Time [sec]")
    plt.ylabel("Z")
    plt.title("2 Node ANCF Beam Mesh - Tip Z Position")
    plt.show()