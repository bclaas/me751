from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

def _cell_type_from_connectivity(conn: np.ndarray, coors: np.ndarray) -> tuple[str, int]:
    """
    Infer XDMF TopologyType and nodes-per-cell.
    """
    conn = np.asarray(conn)
    k = conn.shape[1]
    if k == 2:
        return "Polyline", 2
    if k == 3:
        return "Triangle", 3
    if k == 8:
        return "Hexahedron", 8
    if k == 4:
        # Distinguish Quad vs Tet: check coplanarity on a sample of cells
        # volume ~ det(p1-p0, p2-p0, p3-p0)/6; near-zero -> planar => Quad
        pts = np.asarray(coors, dtype=np.float64)
        idx = conn[: min(len(conn), 64)]  # sample up to 64
        vols = []
        for a,b,c,d in idx:
            p0, p1, p2, p3 = pts[[a,b,c,d]]
            v = np.linalg.det(np.c_[p1-p0, p2-p0, p3-p0]) / 6.0
            vols.append(abs(v))
        if np.nanmedian(vols) < 1e-14 * np.max(np.linalg.norm(pts, axis=1) + 1.0):
            return "Quadrilateral", 4
        return "Tetrahedron", 4
    raise ValueError(f"Unsupported connectivity with {k} nodes per element; expected 3,4, or 8.")

def write_xlsx(results : dict, out_path: str | Path):
    out_path = Path(out_path).with_suffix(".xlsx")
    
    header = ["Time"]
    body_names = [k for k,_ in results.items() if k!="time"]
    for k in body_names: header += [f"{k} x", f"{k} y", f"{k} z", f"{k} e0", f"{k} e1", f"{k} e2", f"{k} e3"]
    
    assert len(results["time"]) == len(results[body_names[0]]["Results"])

    data = []
    for ii in range(len(results["time"])):
        row = []
        row.append(results["time"][ii])
        for body_name in body_names:
            for foo in results[body_name]["Results"][ii]:   # Lazy. oh well.
                row.append(foo)
        
        data.append(row)
    
    df = pd.DataFrame(data=data, columns=header)
    df.set_index("Time", inplace=True)
    df.to_excel(out_path)

def write_xdmf(results: dict, out_path: str | Path):
    """
    Convert a rigid-body multi-body result into XDMF v3 + HDF5.

    Parameters
    ----------
    results : dict
        See your 'results_format.txt':
        - results["time"] -> list of floats (length T)
        - for each body name:
          results[body]["Geometry"]["Connectivity"] : (Nc, k) int array (0-based indices)
          results[body]["Geometry"]["Coors"]        : (Np, 3) float array (node coords w.r.t. CG)
          results[body]["Geometry"]["Datum"]           : (3,) float array (initial CG in global)
          results[body]["Results"]                  : list of length T, each q := [dx,dy,dz, e0,e1,e2,e3]
    out_path : str | Path
        Target .xdmf file. The .h5 will be placed alongside unless h5_path is given.
    """
    out_path = Path(out_path)
    out_path = out_path.parent / out_path.stem
    h5_path = out_path.with_suffix(".h5")
    xdmf_path = out_path.with_suffix(".xdmf")

    times = np.asarray(results["time"], dtype=np.float64)
    assert len(times) > 0

    with h5py.File(h5_path, "w") as h5:
        # write once-per-body connectivity; write per-step coordinates
        bodies = [k for k in results.keys() if k != "time"]
        meta = {}  # per-body metadata for XDMF assembly

        for body in bodies:
            g = results[body]["Geometry"]
            conn = np.asarray(g["Connectivity"], dtype=np.int32)
            pts_local = np.asarray(g["Coors"], dtype=np.float64)
            datum = np.asarray(g["Datum"], dtype=np.float64).ravel()  # Geometry's initial CG in G-RF 
            if pts_local.shape[1] != 3 or datum.size != 3:
                raise ValueError(f"{body}: expect 3D coordinates")

            topo_type, k = _cell_type_from_connectivity(conn, pts_local)
            Nc = conn.shape[0]
            Np = pts_local.shape[0]

            # HDF5 groups
            grp_topo = h5.require_group(f"topology/{body}")
            dset_conn = grp_topo.create_dataset("connectivity", data=conn, dtype=np.int32, compression="gzip")

            grp_pts = h5.require_group(f"points/{body}")

            # write coordinates for each time step
            qs = results[body]["Results"]
            if len(qs) != len(times):
                raise ValueError(f"{body}: Results length {len(qs)} != time length {len(times)}")

            for ti, q in enumerate(qs):
                q = np.asarray(q, dtype=np.float64).ravel()
                if q.size != 7:
                    raise ValueError(f"{body}: each q must have 7 entries [dx,dy,dz,q0,q1,q2,q3]")
                dr = q[:3]
                [e0, e1, e2, e3] = q[3:]
                # Re-order per SciPy convention
                quat = [e1, e2, e3, e0]
                quat = quat / np.linalg.norm(quat)
                R = Rotation.from_quat(quat).as_matrix()
                #pts_t = pts_local @ R.T  # rotate local -> global frame
                pts_t = np.array([(R @ (datum + ploc)) + dr for ploc in pts_local])
                #pts_t += (datum + dr)      # translate datum + delta r(t)

                # Keep
                dset = grp_pts.create_dataset(
                    f"t{ti:05d}", data=pts_t.astype(np.float64),
                    compression="gzip"
                )

            meta[body] = dict(
                topo_type=topo_type,
                nodes_per_cell=k,
                Nc=Nc,
                Np=Np,
                conn_path=f"{h5_path.name}:{dset_conn.name}",  # "file.h5:/topology/Body/connectivity"
                # points path is formatted per-time below
            )

    # build XDMF v3
    def dataitem_text(dim_str, number_type, precision, h5_relpath):
        return ET.Element(
            "DataItem",
            dict(Dimensions=dim_str, NumberType=number_type,
                 Precision=str(precision), Format="HDF")
        ), h5_relpath

    X = ET.Element("Xdmf", dict(Version="3.0"))
    domain = ET.SubElement(X, "Domain")
    coll_time = ET.SubElement(domain, "Grid", dict(Name="AllBodies", GridType="Collection", CollectionType="Temporal"))

    for ti, tval in enumerate(times):
        coll_space = ET.SubElement(coll_time, "Grid",
                                   dict(Name=f"t={tval:g}", GridType="Collection", CollectionType="Spatial"))
        ET.SubElement(coll_space, "Time", dict(Type="Single", Value=f"{tval:g}"))

        for body, m in meta.items():
            G = ET.SubElement(coll_space, "Grid", dict(Name=body, GridType="Uniform"))

            # Topology
            topo = ET.SubElement(G, "Topology", dict(TopologyType=m["topo_type"], NumberOfElements=str(m["Nc"])))
            di_topo, text_topo = dataitem_text(f'{m["Nc"]} {m["nodes_per_cell"]}', "Int", "4", m["conn_path"])
            di_topo.text = text_topo
            topo.append(di_topo)

            # Geometry
            geom = ET.SubElement(G, "Geometry", dict(GeometryType="XYZ"))
            points_path = f'{h5_path.name}:/points/{body}/t{ti:05d}'
            di_geom, text_geom = dataitem_text(f'{m["Np"]} 3', "Float", '8', points_path)
            di_geom.text = text_geom
            geom.append(di_geom)

    # pretty print to file
    tree = ET.ElementTree(X)
    _indent_xml(X)
    tree.write(xdmf_path, encoding="utf-8", xml_declaration=True)

def _indent_xml(elem, level: int = 0) -> None:
    # small pretty-printer for ElementTree (so the .xdmf is readable)
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            _indent_xml(e, level+1)
            if not e.tail or not e.tail.strip():
                e.tail = i + "  "
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


if __name__ == "__main__":
    # Test
    import meshzoo
    points, cells = meshzoo.icosa_sphere(4)
    points = points.astype(float)
    cells = cells.astype(np.int32)

    times = np.arange(0.0, 1.0, 0.01)
    z = np.abs(np.cos(6*np.pi*times))

    qs = []
    for zi in z:
        q = np.zeros(7)
        q[2] = zi
        q[-1] = 1
        qs.append(q)

    results = {
        "time": times,
        "ball": {
            "Geometry": {
                "Connectivity": cells,
                "Coors": points,
                "Datum": np.array([0,0,0])
            },
            "Results": qs
        }
    }
    
    write_results(results, "C:/ME751/bouncing_ball")