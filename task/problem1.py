import matplotlib.pyplot as plt

# a: period1開始時の利子を除く資産格子 (shape=(NA,))
# aplus[0,:,:]: period1→period2 の政策関数 (shape=(NA,NL))
# NL=3, 各列が Low, Mid, High の順

# plot_policy.py
import matplotlib.pyplot as plt
from calc_model import solve_three_period_model

def plot_saving_policy():
    # 計算モジュールから結果を取得
    a, aplus = solve_three_period_model()

    labels = ['Low', 'Mid', 'High']
    markers = ['o', 's', '^']
    a_l, a_u = a[0], a[-1]

    plt.figure(figsize=(8,6))
    for il in range(3):
        plt.plot(
            a,
            aplus[0, :, il],
            marker=markers[il],
            linestyle='-',
            label=labels[il]
        )

    plt.xlabel('first term asset')
    plt.ylabel('second term asset')
    plt.title('policy function with productivity')
    plt.legend()
    plt.grid(True)
    plt.xlim(a_l, a_u)
    plt.ylim(a_l, a_u)
    plt.show()

if __name__ == "__main__":
    plot_saving_policy()
