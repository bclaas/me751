import numpy as np
import pandas as pd

def newton_raphson(f, fprime, x0, iters=15):
    count = 0
    df = pd.DataFrame(columns=["k"])
    df["k"] = [ii for ii in range(iters)]
    df.set_index("k", inplace=True)
    x = x0
    while count < iters:
        fx = f(x)
        df.at[count, "x"] = x
        df.at[count, "f(x)"] = fx

        x = x - fx/fprime(x)
        count += 1

    return df

if __name__ == "__main__":
    f = lambda x: x**6 - x - 1
    fprime = lambda x: 6*x**5 - 1
    a = 1.13472413840152
    x0 = -1

    df = newton_raphson(f, fprime, x0)
    df["x-a"] = df["x"] - a
    xkp1 = np.delete(df["x"].to_numpy(), 0); xkp1 = np.append(xkp1, 0)
    df["x(k+1) - x(k)"] = xkp1 - df["x"]
    print(df)



