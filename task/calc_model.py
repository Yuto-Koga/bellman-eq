#file for function, parameter
# importing libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# calc_model.py

import numpy as np

import numpy as np

# utility function with lower bound to avoid numerical issues
def util(cons, gamma):
    c = max(cons, 1e-4)
    return c**(1.0-gamma) / (1.0-gamma)

# solve three-period model with income risk
def solve_three_period_model(
    gamma=2.0,
    beta=0.985**20,
    r=1.025**20 - 1.0,
    y=np.array([1.0, 1.2, 0.4]),
    l=np.array([0.8027, 1.0, 1.2457]),
    prob=np.array([
        [0.7451, 0.2528, 0.0021],
        [0.1360, 0.7281, 0.1361],
        [0.0021, 0.2528, 0.7451]
    ]),
    a_l=0.0,
    a_u=2.0,
    NA=100
):
    """
    Returns:
      a      : asset grid (size NA)
      aplus : policy function array (JJ x NA x NL)
    """
    # settings
    JJ = 3
    NL = l.size

    # asset grid
    a = np.linspace(a_l, a_u, NA)

    # initialize arrays
    v = np.zeros((JJ, NA, NL))
    iaplus = np.zeros((JJ, NA, NL), dtype=int)
    aplus = np.zeros((JJ, NA, NL))

    # --- period 3: terminal period value ---
    for ia in range(NA):
        v[2, ia, :] = util(y[2] + (1.0 + r)*a[ia], gamma)

    # --- period 2: no uncertainty in next period utility index 0 ---
    for il in range(NL):
        for ia in range(NA):
            reward = np.zeros(NA)
            for iap in range(NA):
                reward[iap] = (
                    util(l[il] + (1.0 + r)*a[ia] - a[iap], gamma)
                    + beta * v[2, iap, 0]
                )
            opt = np.argmax(reward)
            iaplus[1, ia, il] = opt
            aplus[1, ia, il] = a[opt]
            v[1, ia, il] = reward[opt]

    # --- period 1: with income risk ---
    for il in range(NL):
        for ia in range(NA):
            reward = np.zeros(NA)
            for iap in range(NA):
                EV = np.sum(prob[il] * v[1, iap, :])
                reward[iap] = (
                    util(l[il] + (1.0 + r)*a[ia] - a[iap], gamma)
                    + beta * EV
                )
            opt = np.argmax(reward)
            iaplus[0, ia, il] = opt
            aplus[0, ia, il] = a[opt]
            v[0, ia, il] = reward[opt]

    return a, aplus

if __name__ == "__main__":
    a, aplus = solve_three_period_model()
    print("asset grid a (first, last):", a[0], a[-1])
    print("policy function shape:", aplus.shape)
