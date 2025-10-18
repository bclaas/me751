import numpy as np

def vec2quat(theta, u: np.ndarray):
    u = u / np.linalg.norm(u)
    e0 = np.cos(theta/2)
    [e1, e2, e3] = np.sin(theta/2) * u
    return np.array([e0, e1, e2, e3])


def quat2A(e0, e1, e2, e3):
    return 2*np.array([[e0**2 + e1**2 - 0.5, e1*e2 - e0*e3, e1*e3 - e0*e2],
                  [e1*e2 + e0*e3, e0**2 + e2**2 - 0.5, e2*e3 - e0*e1],
                  [e1*e3 - e0*e2, e2*e3 + e0*e1, e0**2 + e3**2 - 0.5]])


if __name__ == "__main__":
    theta = np.pi / 3
    u = np.array([1/3, 2/3, -2/3])
    r = np.array([1, -2, 5])

    q = vec2quat(theta, u)
    [e0, e1, e2, e3] = q
    A = quat2A(e0, e1, e2, e3)

    sP = np.array([1, 0, 1])
    sP_loc = A.T @ sP - r
    print(f"{sP_loc = }")

