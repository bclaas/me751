import numpy as np
import matplotlib.pyplot as plt

def get_hilbert_matrix(n) -> np.ndarray:
    H = [[i + j - 1 for j in range(1, n+1)] for i in range(1, n+1)]
    H = 1. / np.array(H)
    return H

if __name__ == "__main__":
    # Part a
    H = get_hilbert_matrix(10)
    cn = np.linalg.cond(H)
    print(f"{cn = }")

    # Part b
    b0 = np.ones(10)
    x0 = np.linalg.solve(H, b0)
    b1 = b0 + 0.0001
    x1 = np.linalg.solve(H, b1)
    x_change = np.linalg.norm(x1-x0, ord=2) / np.linalg.norm(x0, ord=2)
    b_change = np.linalg.norm(b1-b0, ord=2) / np.linalg.norm(b0, ord=2)
    print(f"{x_change = }")
    print(f"{b_change = }")
    
    # Part c
    fig = plt.figure()
    orders = [ii for ii in range(10, 16)]
    cns = [np.linalg.cond(get_hilbert_matrix(ii)) for ii in orders]
    plt.scatter(orders, cns)
    plt.grid()
    plt.xlabel("Hilbert Order")
    plt.ylabel("Condition Number")
    plt.yscale('log')
    print(orders)
    print(cns)
    plt.show()