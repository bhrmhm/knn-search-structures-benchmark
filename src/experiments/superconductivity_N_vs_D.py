# N vs D effect on query time - superconductivity dataset
#Heads up: it can take so much time

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from src.KNNClassifier import bestLeafSize, knnClassifier_tree
from src.plots import _3D_plot, plot_heatmaps

#-----Main-----
#fetch data
superconductivity = fetch_ucirepo(id=464)
X_full = superconductivity.data.features.values
y_full = superconductivity.data.targets.values.ravel()
#it's for regression so convert to binary classification
y_full = (y_full > np.median(y_full)).astype(int)  # above/below median

# standardize full data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
#------------

#For superconductivity dataset
N_values = [1000,2000,5000, 8000, 10000, 15000,20000,20500]
D_values = [2,5,10, 15, 20, 30, 50, 81]



best_leaf_kd   = np.zeros((len(D_values), len(N_values)), dtype=int)
best_leaf_ball = np.zeros((len(D_values), len(N_values)), dtype=int)

query_times = {'kd_tree': [], 'ball_tree': [], 'brute': []}
per_point_times   = {'kd_tree': [], 'ball_tree': [], 'brute': []}

query_times_default = {'kd_tree': [], 'ball_tree': [], 'brute': []}
per_point_times_default  = {'kd_tree': [], 'ball_tree': [], 'brute': []}

test_size = 500
valid_size = 200

K = 5
# iterate every combination of N and D
for i, d in enumerate(D_values):                          # outer loop: var_list2

    # Reduce dimensions with PCA
    X_d = PCA(n_components=d).fit_transform(X_scaled)


    X_fixed = X_d[:test_size + valid_size]
    y_fixed = y_full[:test_size + valid_size]
    Xvalid_fixed = X_fixed[:valid_size]
    yvalid_fixed = y_fixed[:valid_size]
    Xtest_fixed = X_fixed[valid_size:]
    ytest_fixed = y_fixed[valid_size:]
    for j, n in enumerate(N_values):                      # inner loop: var_list1
        print(f"N={n}, D={d}")

        # take n samples
        X_train = X_d[test_size + valid_size:
                      test_size + valid_size + n]
        y_train = y_full[test_size + valid_size:
                         test_size + valid_size + n]

        if len(X_train) < n:
            print(f"  Not enough data for N={n} at D={d}, skipping")
            continue

        print(f"train={len(X_train)}, valid={len(Xvalid_fixed)}, test={len(Xtest_fixed)}")


        #Tune best leaf size on validation set
        best_leaf_kd[i, j] = bestLeafSize(X_train, Xvalid_fixed, y_train, K, 10, 100, 'kd_tree')
        best_leaf_ball[i, j] = bestLeafSize(X_train, Xvalid_fixed, y_train, K, 10, 100, 'ball_tree')


        results_KD = knnClassifier_tree(X_train, y_train, Xtest_fixed,K, algo ="kd_tree", leaf_size=best_leaf_kd[i, j], n_repeats=10)
        results_Ball = knnClassifier_tree(X_train, y_train, Xtest_fixed,K, algo ="ball_tree", leaf_size=best_leaf_ball[i, j], n_repeats=10)
        results_Brute = knnClassifier_tree(X_train, y_train, Xtest_fixed,K, algo ="brute", leaf_size=30, n_repeats=10)

        # measure with DEFAULT leaf size (30)
        res_kd_def = knnClassifier_tree(X_train, y_train, Xtest_fixed, K,
                                        algo='kd_tree',
                                        leaf_size=30, n_repeats=10)
        res_ball_def = knnClassifier_tree(X_train, y_train, Xtest_fixed, K,
                                          algo='ball_tree',
                                          leaf_size=30, n_repeats=10)


        for algo, res in [('kd_tree', results_KD),
                          ('ball_tree', results_Ball),
                          ('brute', results_Brute)]:
            query_times[algo].append(res['query'])
            per_point_times[algo].append(res['time_per_point'])

        for algo, res in [('kd_tree', res_kd_def),
                          ('ball_tree', res_ball_def),
                          ('brute', results_Brute)]:
            query_times_default[algo].append(res['query'])
            per_point_times_default[algo].append(res['time_per_point'])


#speedup of using best Leaf size to default
print("--- Speedup: best leaf size vs default leaf size---")
print("N | D | KD speedup | Ball speedup")
print("-" * 50)
kd_speedups = []
ball_speedups = []
nb_col = len(N_values)
for i, d in enumerate(D_values):
    for j,n in enumerate(N_values):
        idx = i * nb_col + j
        kd_speedup = query_times_default['kd_tree'][idx] / query_times['kd_tree'][idx]
        ball_speedup = query_times_default['ball_tree'][idx] / query_times['ball_tree'][idx]

        kd_speedups.append(kd_speedup)
        ball_speedups.append(ball_speedup)

        print(f"N={n} | D={d} |{kd_speedup:.2f}x faster | {ball_speedup:.2f}x faster")

# plot
_3D_plot(N_values, D_values, "best_leaf_size",query_times,xlabel='N', ylabel='D',title='KNN Performance: N vs D (with best leaf size)')
_3D_plot(N_values, D_values, "best_leaf_size_whBT",query_times,xlabel='N', ylabel='D',title='KNN Performance: N vs D (with best leaf size, without BallT)', has_BT=False)

_3D_plot(N_values, D_values, "default_leaf_size",query_times_default,xlabel='N', ylabel='D',title='KNN Performance: N vs D (Default leaf size)')



#Query time heatmap
plot_heatmaps(N_values, D_values, query_times, xlabel='N', ylabel='D', value_format='.4f', colorbar_label='seconds',title='Query Time', has_brute=True)

# Best leaf size heatmap
best_leaf = {
    'kd_tree':   best_leaf_kd.flatten().tolist(),
    'ball_tree': best_leaf_ball.flatten().tolist()
}

plot_heatmaps(N_values, D_values, best_leaf, xlabel='N', ylabel='D',value_format='d',colorbar_label='Leaf Size',title='Best Leaf Size')

# Speedup heatmaps
speedup_dict = {
    'kd_tree':   kd_speedups,
    'ball_tree': ball_speedups
}

plot_heatmaps(N_values, D_values, speedup_dict,
              xlabel='N', ylabel='D',
              value_format='.2f',
              colorbar_label='Speedup (x faster)',
              title='Speedup: Best Leaf Size vs Default (30)')

