import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=140, precision=8)

# Global
L = 0.5
W = 0.003
H = 0.003
rho = 7700.0
E_mod = 2.0e11
nu = 0.3
lam = E_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E_mod / (2.0 * (1.0 + nu))
gvec = np.array([0.0, 0.0, -9.81])


def bvec(u, v, w):
    return np.array([1.0, u, v, w, u * v, u * w, u * u, u ** 3], dtype=float)

def bu(u, v, w):
    return np.array([0.0, 1.0, 0.0, 0.0, v, w, 2.0 * u, 3.0 * u * u], dtype=float)

def bv(u, v, w):
    return np.array([0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0, 0.0], dtype=float)

def bw(u, v, w):
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0], dtype=float)

def make_Binv():
    u1, v1, w1 = -L / 2.0, 0.0, 0.0
    u2, v2, w2 = +L / 2.0, 0.0, 0.0
    B = np.column_stack([
        bvec(u1, v1, w1),
        bu(u1, v1, w1),
        bv(u1, v1, w1),
        bw(u1, v1, w1),
        bvec(u2, v2, w2),
        bu(u2, v2, w2),
        bv(u2, v2, w2),
        bw(u2, v2, w2),
    ])
    return np.linalg.inv(B)

def shape_and_grads(Binv, u, v, w):
    s = Binv @ bvec(u, v, w)
    su = Binv @ bu(u, v, w)
    sv = Binv @ bv(u, v, w)
    sw = Binv @ bw(u, v, w)
    return s, su, sv, sw

def triple_gauss(nu_g, nv_g, nw_g):
    xu, wu = leggauss(nu_g)
    xv, wv = leggauss(nv_g)
    xw, ww = leggauss(nw_g)
    return xu, wu, xv, wv, xw, ww

def integrate_mass(Binv, nu_g=4, nv_g=2, nw_g=2):
    xu, wu, xv, wv, xw, ww = triple_gauss(nu_g, nv_g, nw_g)
    J = (L * W * H) / 8.0
    mbar = np.zeros((8, 8))
    for i, xi in enumerate(xu):
        u = 0.5 * L * xi
        wi = wu[i]
        for j, xj in enumerate(xv):
            v = 0.5 * W * xj
            wj = wv[j]
            for k, xk in enumerate(xw):
                w = 0.5 * H * xk
                wk = ww[k]
                s, _, _, _ = shape_and_grads(Binv, u, v, w)
                wgt = rho * J * wi * wj * wk
                mbar += wgt * np.outer(s, s)
    M = np.kron(mbar, np.eye(3))
    return mbar, M

def gravity_generalized_force(Binv, nu_g=4, nv_g=2, nw_g=2):
    xu, wu, xv, wv, xw, ww = triple_gauss(nu_g, nv_g, nw_g)
    J = (L * W * H) / 8.0
    mg = np.zeros(8)
    for i, xi in enumerate(xu):
        u = 0.5 * L * xi
        wi = wu[i]
        for j, xj in enumerate(xv):
            v = 0.5 * W * xj
            wj = wv[j]
            for k, xk in enumerate(xw):
                w = 0.5 * H * xk
                wk = ww[k]
                s, _, _, _ = shape_and_grads(Binv, u, v, w)
                wgt = rho * J * wi * wj * wk
                mg += wgt * s
    f = np.zeros(24)
    for a in range(8):
        f[3 * a:3 * a + 3] = mg[a] * gvec
    return f

def internal_force_SVK(Binv, N_nodes, nu_g=5, nv_g=3, nw_g=3):
    xu, wu, xv, wv, xw, ww = triple_gauss(nu_g, nv_g, nw_g)
    J = (L * W * H) / 8.0
    f = np.zeros((3, 8))
    I3 = np.eye(3)
    for i, xi in enumerate(xu):
        u = 0.5 * L * xi
        wi = wu[i]
        for j, xj in enumerate(xv):
            v = 0.5 * W * xj
            wj = wv[j]
            for k, xk in enumerate(xw):
                w = 0.5 * H * xk
                wk = ww[k]
                s, su, sv, sw = shape_and_grads(Binv, u, v, w)
                ru = N_nodes @ su
                rv = N_nodes @ sv
                rw = N_nodes @ sw
                F = np.column_stack((ru, rv, rw))
                C = F.T @ F
                Estrain = 0.5 * (C - I3)
                S2 = lam * np.trace(Estrain) * I3 + 2.0 * mu * Estrain
                P = F @ S2
                wgt = J * wi * wj * wk
                for a in range(8):
                    f[:, a] += wgt * (su[a] * P[:, 0] + sv[a] * P[:, 1] + sw[a] * P[:, 2])
    fout = np.zeros(24)
    for a in range(8):
        fout[3 * a:3 * a + 3] = f[:, a]
    return fout

def position_from_uvwt(Binv, N_nodes, u, v, w):
    s, _, _, _ = shape_and_grads(Binv, u, v, w)
    return N_nodes @ s

def tip_force_generalized(Binv, N_nodes, f_tip, u, v, w):
    s, _, _, _ = shape_and_grads(Binv, u, v, w)
    f = np.zeros(24)
    for a in range(8):
        f[3 * a:3 * a + 3] = s[a] * f_tip
    return f

if __name__ == "__main__":
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

    Binv = make_Binv()

    mij, M = integrate_mass(Binv, nu_g=4, nv_g=2, nw_g=2)
    print("mij (8x8) using GQ(nu=4,nv=2,nw=2):")
    print(mij)
    print("Mass matrix M shape:", M.shape)

    f_grav = gravity_generalized_force(Binv, nu_g=4, nv_g=2, nw_g=2)
    print("Generalized force due to gravity (size 24):")
    print(f_grav)

    f_int = internal_force_SVK(Binv, N, nu_g=5, nv_g=3, nw_g=3)
    print("Generalized internal force (SVK) at the given state (size 24):")
    print(f_int)

    u_line = np.linspace(-L / 2.0, L / 2.0, 301)
    curve = np.array([position_from_uvwt(Binv, N, u, 0.0, 0.0) for u in u_line])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Beam Axis (v=0,w=0)')
    plt.tight_layout()
    plt.savefig("beam_axis.png", dpi=160)
    plt.close(fig)
    print("Saved plot to beam_axis.png")

    ang = np.deg2rad(45.0)
    f_tip = np.array([10.0 * np.cos(ang), 0.0, 10.0 * np.sin(ang)])
    f_tip_gen = tip_force_generalized(Binv, N, f_tip, L / 2.0, 0.0, 0.0)
    print("Generalized external tip force (size 24):")
    print(f_tip_gen)

    print("GQ used: mass/gravity nu=4,nv=2,nw=2; internal nu=5,nv=3,nw=3")
    print("Mass matrix is constant in TL and computed once; forces depend on state.")
