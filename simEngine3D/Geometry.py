import numpy as np
import meshzoo
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def rot_a_to_b(b, a=np.array([0,0,1]), eps=1e-12):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        raise ValueError("Rotation undefined for zero-length vector.")
    ah, bh = a/na, b/nb

    v = np.cross(ah, bh)
    c = float(np.dot(ah, bh))
    s2 = np.dot(v, v)

    if s2 > eps**2:  # general case
        K = np.array([[    0, -v[2],  v[1]],
                      [ v[2],     0, -v[0]],
                      [-v[1],  v[0],    0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s2)
        return R

    if c > 0:        # vectors already aligned
        return np.eye(3)

    # opposite directions: 180deg about axis a
    t = np.array([1.0, 0.0, 0.0]) if abs(ah[0]) <= min(abs(ah[1]), abs(ah[2])) \
        else (np.array([0.0, 1.0, 0.0]) if abs(ah[1]) <= abs(ah[2]) else np.array([0.0, 0.0, 1.0]))
    u = np.cross(ah, t); u /= np.linalg.norm(u)
    R = 2.0 * np.outer(u, u) - np.eye(3)
    return R

class Link:
    def __init__(self, p1: np.ndarray, p2: np.ndarray, radius: float=None, n: int=16):
        d = p2 - p1
        L = np.linalg.norm(d)

        if not radius:
            radius = 0.025*L

        coors, self.conn = meshzoo.tube(length=L, radius=radius, n=n)
        coors[:,2] += L/2   # tube axis now extends from (0,0,0) to (0,0,L)...

        rot = rot_a_to_b(d, np.array([0,0,L]))
        for ii in range(len(coors)):
            #coors[ii,:] = rot @ coors[ii,:] + p1
            coors[ii,:] = rot @ coors[ii,:]

        # test_point = np.array([0,0,L])
        # t2 = rot @ test_point + p1
        # assert np.linalg.norm(t2 - p2) < 1e-4
        
        self.coors = coors


def plot_shell(coords, conn, *, filled=False, ax=None):
    C = np.asarray(coords, float)
    # conn can be (M,3)/(M,4) ndarray or an iterable of index arrays
    faces = [C[np.asarray(f, int)] for f in (conn if hasattr(conn, "__iter__") and conn is not None else [conn])]

    ax = ax or plt.figure().add_subplot(projection="3d")
    coll = Poly3DCollection(faces, linewidths=0.4)
    if filled:
        coll.set_alpha(0.7)           # light translucency
    else:
        coll.set_facecolor((0,0,0,0)) # transparent faces
        coll.set_edgecolor('k')
    ax.add_collection3d(coll)

    # equal-axis box
    mn, mx = C.min(0), C.max(0)
    r = (mx - mn).max()
    c = (mn + mx) * 0.5
    ax.set_xlim(c[0]-r/2, c[0]+r/2)
    ax.set_ylim(c[1]-r/2, c[1]+r/2)
    ax.set_zlim(c[2]-r/2, c[2]+r/2)
    return ax


if __name__ == "__main__":
    n0 = np.array([0,0,0])
    n1 = np.array([1,0,0])
    foo = Link(n0, n1)


    # Example:
    ax = plot_shell(foo.coors, foo.conn)
    ax.scatter(n0[0], n0[1], n0[2], color="red", label="n0")
    ax.scatter(n1[0], n1[1], n1[2], color="blue", label="n1")
    plt.show() 
