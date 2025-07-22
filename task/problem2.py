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
a_u = 3
NA = 100
a = np.linspace(a_l, a_u, NA)

# initialization
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))

# backward induction
# calculate tax revenue
tax_vector = 0.3*l
taxrevenue = np.sum(mu_2*tax_vector)
Pension = (1.0+r)*taxrevenue 
print(f"Pension (一人当たりの年金額) = {Pension:.4f}")
# period 3
for ia in range(NA):
    v[2, ia, :] = util(y[2] + (1.0+r)*a[ia] + Pension, gamma)


# period 2
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            reward[iap] = util(0.7*l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*v[2, iap, 0]
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

# t=2 → t=3 の政策関数を例に
np.any(aplus[1,:,:] == a_u)

# t=2→t=3 の政策関数がグリッド上限 a_u に張り付いているかどうか
mask = (aplus[1, :, :] == a_u)

# ブール値を表示
print("張り付きあり？", np.any(mask))       # いずれか True があれば True
print("状態ごとのマスク：\n", mask)         # 各 (資産, 状態) ごとに True/False

# もしどこが張り付いているか知りたければ
idx = np.column_stack(np.where(mask))
print("張り付いているインデックス (ia, state)：\n", idx)

plt.figure()
plt.plot(a, aplus[0, :, 0], marker='o', label='Low')
plt.plot(a, aplus[0, :, 1], marker='s', label='Mid')
plt.plot(a, aplus[0, :, 2], marker='^', label='High')
plt.title("policy function")
plt.xlabel("first term asset")
plt.ylabel("second term asset")
plt.ylim(a_l, a_u)
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(a, aplus[1, :, 0], marker='o', label='Low')
plt.plot(a, aplus[1, :, 1], marker='s', label='Mid')
plt.plot(a, aplus[1, :, 2], marker='^', label='High')
plt.title("policy function")
plt.xlabel("second term asset")
plt.ylabel("third term asset")
plt.ylim(a_l, a_u)
plt.grid(True)
plt.legend()
plt.show()