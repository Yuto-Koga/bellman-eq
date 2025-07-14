# plot_policy.py
import matplotlib.pyplot as plt
from calc_model import solve_three_period_model

def plot_saving_policy():
    # 計算モジュールから結果を取得
    # a.shape = (NA,)
    # aplus.shape = (3, NA, NL)  # 0:period1→2, 1:period2→3, 2:period3→4(最終期の政策)
    a, aplus = solve_three_period_model()
    
    labels_prod = ['Low', 'Mid', 'High']
    markers_prod = ['o', 's', '^']
    linestyles_period = ['-', '--', ':']
    period_labels = ['Period1', 'Period2', 'Period3']
    
    plt.figure(figsize=(10, 6))
    
    # 各期 (t → t+1) ごとにループ
    for iperiod in range(3):
        for il in range(3):
            plt.plot(
                a,
                aplus[iperiod, :, il],
                marker=markers_prod[il],
                linestyle=linestyles_period[iperiod],
                label=f'{period_labels[iperiod]} / {labels_prod[il]}'
            )
    
    plt.xlabel('期初資産 $a_t$（利子を除く）')
    plt.ylabel('次期資産 $a_{t+1}$（利子を除く）')
    plt.title('各期・各生産性別 貯蓄政策関数の比較')
    plt.legend(loc='upper left', fontsize='small', ncol=2)
    plt.grid(True)
    plt.xlim(a.min(), a.max())
    plt.ylim(a.min(), a.max())
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_saving_policy()
