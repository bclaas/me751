import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Link:
    def __init__(self, L, radius: float=None, n: int=8):
        if not radius:
            radius = 0.025*L

        n_slices = 4
        coors0 = []
        xs = np.linspace(0, L, n_slices)
        for theta in np.linspace(0, 2*np.pi, n+1):
            for xi in xs:
                coors0.append([xi, theta])

        coors0 = np.array(coors0)
        mesh = Delaunay(coors0)
        conn = mesh.simplices

        last_idxs = [len(coors0)-ii-1 for ii in reversed(range(n_slices))] # Indexes of nodes in last column
        first_idxs = [ii for ii in range(n_slices)]
        for target, replace_with in zip(last_idxs, first_idxs):
            mask = (conn == target)
            conn[mask] = replace_with
        
        # Map planar coordinates around cylinder
        coors = []
        for [xi, theta] in coors0:
            yi = radius*np.cos(theta)
            zi = radius*np.sin(theta)
            coors.append([xi, yi, zi])
        
        coors = np.array(coors)
        self.coors = coors
        self.conn = conn
        
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
    # Example:
    foo = Link(L=1, radius=0.1)
    ax = plot_shell(foo.coors, foo.conn)
    plt.show()