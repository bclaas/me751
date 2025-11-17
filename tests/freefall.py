import numpy as np
from copy import deepcopy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from simEngine3D.Assembly import Assembly
from simEngine3D.Orientation import Orientation, A_to_p
from simEngine3D.Bodies import RigidBody
from simEngine3D.KCons import DP1, DP2, CD, D
#from simEngine3D.Post import write_xdmf, write_xlsx
from simEngine3D.Post import write_xlsx
#from simEngine3D.Geometry import Link
from simEngine3D.solvers import run_dynamics_hht, run_dynamics

def vecs2ori(f, g, h):
    A = np.zeros((3, 3))
    A[:, 0] = f
    A[:, 1] = g
    A[:, 2] = h
    e0, e1, e2, e3 = A_to_p(A)
    return Orientation(e0, e1, e2, e3)


if __name__ == "__main__":
    # Ball
    r1 = np.array([0,0,0])
    f, g, h = np.array([1, 0, 0]), np.array([0,1,0]), np.array([0,0,1])
    ori1 = vecs2ori(f,g,h)
    ball = RigidBody("Ball", r1, ori1)
    ball.mass = 1.0
    ball.inertia = np.diag([1, 1, 1])

    # Assembly
    asy = Assembly()
    asy.add_body(ball)

    # Constraint attachment points in local frames (left and right ends)
    siPbar = np.array([-0.5, 0.0, 0.0])  # left end of body i
    sjQbar = np.array([0.5, 0.0, 0.0])  # right end of body j
    xyz = list("XYZ")

    # for jnt in asy.joints:
    #     siPbar = jnt.siPbar
    #     sjQbar = jnt.sjQbar
    #     rj = jnt.jbody.r
    #     Aj = jnt.jbody.ori.A
    #     ri = jnt.ibody.r
    #     Ai = jnt.ibody.ori.A
    #     dij = rj + (Aj @ sjQbar) - ri - (Ai @ siPbar)
    #     phi = jnt.phi(0)
    #     print(f"{jnt.name}: {dij = }; {phi = }")

    # Gravity
    asy.add_grav(np.array([0.0, 0.0, -9.81]))

    # Solve
    dt = 1.0e-4
    end_time = 1.0
    q_results, times = run_dynamics(asy,
                                    dt=dt,
                                    end_time=end_time,
                                    write_increment=100,
                                    max_inner_its=25,
                                    relaxation=0.5,
                                    error_thres=1.0)

    # Post-process: pack geometry + state history
    results = {
        "time": times,
        "Ball": {
            "Geometry": {
                #"Connectivity": link1_geo.conn,
                #"Coors": link1_geo.coors,
                "Datum": np.array([0, 0.0, 0.0]),
            },
            "Results": q_results
        }
    }

    # Write results (adjust path as needed)
    out_folder = Path(__file__).parent
    write_xlsx(results, out_folder / "freefall")
    #write_xdmf(results, out_folder / "freefall")