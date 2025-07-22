# importing libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# utility function

def util(cons, gamma):
    return max(cons, 1e-4)**(1.0-gamma)/(1.0-gamma)

# 3-period model with income risk

# parameters
gamma = 2.0
beta = 0.985**20
r = 1.025**20-1.0
y = np.array([1.0, 1.2, 0.4])
JJ = 3
l = np.array([0.8027, 1.0, 1.2457])
NL = 3
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1361],
    [0.0021, 0.2528, 0.7451]
])
mu_1 = np.array([1.0/NL, 1.0/NL, 1.0/NL])
mu_2 = np.zeros(NL)

for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il, ilp]*mu_1[il]


# grids
a_l = 0.0
a_u = 5
NA = 100
a = np.linspace(a_l, a_u, NA)

# initialization
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))

# backward induction

# period 3
for ia in range(NA):
    v[2, ia, :] = util(y[2] + (1.0+r)*a[ia], gamma)


# period 2
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*v[2, iap, 0]
        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# period 1
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):

            EV = 0.0
            for ilp in range(NL):
                EV += prob[il, ilp]*v[1, iap, ilp]

            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*EV

        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]

# overall utility
# a グリッドの先頭が 0 なら、そのインデックスを ia0=0 とします
ia0 = 0      
a0  = a[ia0]  # = 0

# 各タイプ il ごとの割引生涯効用を格納
Ou = np.zeros(NL)

for il in range(NL):
    # --- period 1 ---
    i1 = iaplus[0, ia0, il]   # 初期資産ゼロのときの選択インデックス
    a1 = a[i1]                # 期2開始時の資産

    c1 = l[il] + (1+r)*a0 - a1
    u1 = util(c1, gamma)

    # --- period 2 ---
    i2 = iaplus[1, i1, il]
    a2 = a[i2]

    c2 = l[il] + (1+r)*a1 - a2
    u2 = util(c2, gamma)

    # --- period 3 ---
    c3 = y[2] + (1+r)*a2 
    u3 = util(c3, gamma)

    # 割引生涯効用を格納
    Ou[il] = u1 + beta*u2 + beta**2 * u3

# 人口シェア mu_1 で加重平均してスカラー値を得る
E_Ou = np.dot(mu_1, Ou)

print(f"初期資産 a₀=0 のときの全体平均期待生涯効用 E_Ou: {E_Ou:.6f}")
