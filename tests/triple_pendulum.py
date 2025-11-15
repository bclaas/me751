import numpy as np
from copy import deepcopy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from simEngine3D.Assembly import Assembly
from simEngine3D.Orientation import Orientation, A_to_p
from simEngine3D.Bodies import RigidBody
from simEngine3D.KCons import DP1, DP2, CD, D
from simEngine3D.Post import write_xdmf, write_xlsx
from simEngine3D.Geometry import Link
from simEngine3D.solvers import run_dynamics_hht, run_dynamics

def split_sys_q(lst_qsys, asy):
    out = {bdy.name : [] for bdy in asy.bodies}
    for qsys in lst_qsys:
        for bdy in asy.bodies:
            qloc = np.zeros(7)
            qloc[0:3] = qsys[7*bdy._id:7*bdy._id+3]
            qloc[3:7] = qsys[7*bdy._id+3:7*bdy._id+7]
            out[bdy.name].append(qloc)

    return out

def vecs2ori(f, g, h):
    A = np.zeros((3, 3))
    A[:, 0] = f
    A[:, 1] = g
    A[:, 2] = h
    e0, e1, e2, e3 = A_to_p(A)
    return Orientation(e0, e1, e2, e3)


# Triple pendulum
if __name__ == "__main__":
    # Node positions defining three 1.0-long links laid along +X
    n0 = np.array([0, 0, 0])
    n1 = np.array([1, 0, 0])
    n2 = np.array([1, 0, -1])
    n3 = np.array([1, 1, -1])

    # Body reference positions at the CGs
    r1 = 0.5 * (n0 + n1)
    r2 = 0.5 * (n1 + n2)
    r3 = 0.5 * (n2 + n3)

    # Link 1
    f, g, h = np.array([1, 0, 0]), np.array([0,1,0]), np.array([0,0,1])
    ori1 = vecs2ori(f,g,h)
    link1 = RigidBody("Link1", r1, ori1)
    link1.mass = 2.0
    link1.inertia = np.diag([0.01, 0.06, 0.06])  # Jbar at COM (L-RF)
    link1_geo = Link(n0, n1)

    # Link 2
    f, g, h = np.array([0, 0, -1]), np.array([-1,0,0]), np.array([0,1,0])
    ori2 = vecs2ori(f,g,h)
    link2 = RigidBody("Link2", r2, ori2)
    link2.mass = 2.0
    link2.inertia = np.diag([0.01, 0.06, 0.06])
    link2_geo = Link(n1, n2)

    # Link 3
    f, g, h = np.array([0, 1, 0]), np.array([-1,0,0]), np.array([0,0,1])
    ori3 = vecs2ori(f,g,h)
    link3 = RigidBody("Link3", r3, ori3)
    link3.mass = 2.0
    link3.inertia = np.diag([0.01, 0.06, 0.06])
    link3_geo = Link(n2, n3)

    # Assembly
    asy = Assembly()
    asy.add_body(link1)
    asy.add_body(link2)
    asy.add_body(link3)

    # Constraint attachment points in local frames (left and right ends)
    siPbar = np.array([-0.5, 0.0, 0.0])  # left end of body i
    sjQbar = np.array([0.5, 0.0, 0.0])  # right end of body j
    xyz = list("XYZ")

    # Link 1 to ground at origin (spherical via 3 CDs at the left end)
    for idx in range(3):
        name = f"CD01-{xyz[idx]}"
        c = np.zeros(3); c[idx] = 1.0
        cd = CD(c, ibody=link1, siPbar=siPbar, f=0.0, name=name)
        asy.add_joint(deepcopy(cd))

    # Link 1 to Link 2 (point coincidence at Link1 right end ↔ Link2 left end)
    for idx in range(3):
        name = f"CD12-{xyz[idx]}"
        c = np.zeros(3); c[idx] = 1.0
        cd = CD(c, ibody=link2, siPbar=siPbar, jbody=link1, sjQbar=sjQbar, f=0.0, name=name)
        asy.add_joint(deepcopy(cd))

    # Link 2 to Link 3
    for idx in range(3):
        name = f"CD23-{xyz[idx]}"
        c = np.zeros(3); c[idx] = 1.0
        cd = CD(c, ibody=link3, siPbar=siPbar, jbody=link2, sjQbar=sjQbar, f=0.0, name=name)
        asy.add_joint(deepcopy(cd))

    for jnt in asy.joints:
        siPbar = jnt.siPbar
        sjQbar = jnt.sjQbar
        rj = jnt.jbody.r
        Aj = jnt.jbody.ori.A
        ri = jnt.ibody.r
        Ai = jnt.ibody.ori.A
        dij = rj + (Aj @ sjQbar) - ri - (Ai @ siPbar)
        phi = jnt.phi(0)
        print(f"{jnt.name}: {dij = }; {phi = }")

    # Gravity
    asy.add_grav(np.array([0.0, 0.0, -9.81]))

    # Integrate (your current L1/L2 driver function)
    dt = 1.0e-4
    end_time = 5.0
    q_results, times = run_dynamics(asy,
                                    dt=dt,
                                    end_time=end_time,
                                    write_increment=100,
                                    max_inner_its=25,
                                    relaxation=0.5,
                                    error_thres=1.0)

    # Split results by body
    q_split = split_sys_q(q_results, asy)

    # Post-process: pack geometry + state history
    results = {
        "time": times,
        "Link1": {
            "Geometry": {
                "Connectivity": link1_geo.conn,
                "Coors": link1_geo.coors,
                "Datum": np.array([0, 0.0, 0.0]),
            },
            "Results": q_split["Link1"]
        },
        "Link2": {
            "Geometry": {
                "Connectivity": link2_geo.conn,
                "Coors": link2_geo.coors,
                "Datum": np.array([0, 0.0, 0.0]),
            },
            "Results": q_split["Link2"]
        },
        "Link3": {
            "Geometry": {
                "Connectivity": link3_geo.conn,
                "Coors": link3_geo.coors,
                "Datum": np.array([0, 0.0, 0.0]),
            },
            "Results": q_split["Link3"]
        },
    }

    # Write results (adjust path as needed)
    out_folder = Path(__file__).parent
    write_xlsx(results, out_folder / "triple_pendulum")
    write_xdmf(results, out_folder / "triple_pendulum")