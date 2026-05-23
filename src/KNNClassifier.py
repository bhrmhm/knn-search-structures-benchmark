import os
import time

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier, KDTree, BallTree



def knnClassifier(Xtrain, y, k, Xtest, n_repeats=5, leaf_size=30):
    n_test = len(Xtest)
    results ={}
    algos = ["brute", "kd_tree", "ball_tree"]

    #warm-up data
    X_warmup = np.random.rand(2, Xtest.shape[1])  # same D, different points
    X_single = np.random.rand(1, Xtest.shape[1])  # single point, same D


    for algo in algos:
        fit_times = []
        query_times = []
        time_per_points = []
        single_point_times = []
        knn = KNeighborsClassifier(n_neighbors=k, algorithm=algo,leaf_size=leaf_size)  # algorithm: auto, brute, kd_tree, ball_tree
        knn.fit(Xtrain, y)
        knn.predict(X_warmup)  # warm up

        for _ in range(n_repeats):
            #knn = KNeighborsClassifier(n_neighbors=k, algorithm=algo,leaf_size=leaf_size)

            #fit time
            start_time = time.time()
            knn.fit(Xtrain, y)
            fit_times.append(time.time() - start_time)

            #knn.predict(X_warmup)  #warms up
            """Explanation: On the first value of the list (N, D, etc), the knn object is a fresh instance and has a cold start. 
            Initialization overhead dominates the at small N (or D)"""


            #query time
            start_time = time.time()
            y_pred =  knn.predict(Xtest)
            query_t = time.time() - start_time
            query_times.append(query_t)
            time_per_points.append(query_t / n_test) #query time per test point

            # query time on single point
            start_time = time.time()
            knn.predict(X_single)
            single_point_times.append(time.time() - start_time)


        results[algo] = {
            'fit': np.mean(fit_times),
            'query': np.mean(query_times),
            'total': np.mean(fit_times) + np.mean(query_times),
            'time_per_point': np.mean(time_per_points),
            'single_point': np.mean(single_point_times)
        }
    return results





def knnClassifier_direct_build(Xtrain, y, k, Xtest, n_repeats=5, leaf_size=30):
    """In this function we measure only the time for building the tree and the query time
    without measuring the whole Fit process

    tree.query doesn't do a classification, it only find the indices of the k nearest neighbors
    not the majority label"""
    n_test = len(Xtest)
    results ={}
    algos = ["brute", "kd_tree", "ball_tree"]
    # warm-up data
    X_warmup = np.random.rand(2, Xtest.shape[1])
    for algo in algos:
        fit_times = []
        query_times = []
        time_per_points = []
        knn = KNeighborsClassifier(n_neighbors=k, algorithm=algo, leaf_size=leaf_size)
        knn.fit(Xtrain, y)
        knn.predict(X_warmup)  # warms up
        for _ in range(n_repeats):
            #knn.fit(Xtrain, y)
            if algo == "kd_tree":
                #Build the tree - fit time
                start_time = time.time()
                tree = KDTree(Xtrain, leaf_size=leaf_size, metric='euclidean') #default = 40
                fit_times.append(time.time() - start_time)

                #query time
                start_time = time.time()
                #distances, indices = tree.query(Xtest, k)
                '''slices = np.array_split(Xtest, 1)
                parallel_results = Parallel(n_jobs=1, prefer='threads')(
                    delayed(tree.query)(chunk, k=k, return_distance=False)
                    for chunk in slices
                )
                indices = np.vstack(parallel_results)
                neighbor_labels = y[indices]  # label lookup
                y_pred = scipy_mode(neighbor_labels, axis=1)[0].ravel()  # majority vote'''
                knn.predict(Xtest)
                query_t = time.time() - start_time
                query_times.append(query_t)
                time_per_points.append(query_t / n_test)

            if algo == "ball_tree":
                # Build the tree - fit time
                start_time = time.time()
                tree = BallTree(Xtrain, leaf_size=leaf_size, metric='euclidean')  # default = 40
                fit_times.append(time.time() - start_time)

                # query time
                start_time = time.time()
                knn.predict(Xtest)
                query_t = time.time() - start_time
                query_times.append(query_t)
                time_per_points.append(query_t / n_test)

            if algo == "brute":

                #fit time
                start_time = time.time()
                _ = np.array(Xtrain)  # equivalent to brute fit - just storing data
                fit_times.append(time.time() - start_time)

                #query time
                start_time = time.time()
                '''distances = euclidean_distances(Xtest, Xtrain, squared=True)  # squared=True: no sqrt
                indices = np.argpartition(distances, k, axis=1)[:, :k]
                neighbor_labels = y[indices]  # label lookup
                y_pred = scipy_mode(neighbor_labels, axis=1)[0].ravel()  # majority vote'''
                knn.predict(Xtest)
                query_t = time.time() - start_time
                query_times.append(query_t)
                time_per_points.append(query_t / n_test)


        results[algo] = {
            'fit': np.mean(fit_times),
            'query': np.mean(query_times),
            'total': np.mean(fit_times) + np.mean(query_times),
            'time_per_point': np.mean(time_per_points)
        }
    return results


def knnClassifier_tree(Xtrain, ytrain, Xtest, K,algo, leaf_size,  n_repeats=10):
    """Measure query time for a given algo
    algo can be any of these:{'kd_tree','ball_tree', 'brute}"""
    n_test = len(Xtest)

    # warm-up data
    X_warmup = np.random.rand(2, Xtest.shape[1])  # same D, different points


    knn = KNeighborsClassifier(n_neighbors=K, algorithm=algo, leaf_size=leaf_size)
    knn.fit(Xtrain, ytrain)
    knn.predict(X_warmup)  # warm up

    query_times = []

    for _ in range(n_repeats):
        # query time

        start = time.time()
        y_pred = knn.predict(Xtest)
        query_t = time.time() - start
        query_times.append(query_t)

    return {'query': np.mean(query_times),
            'time_per_point': np.mean(query_times)/n_test,
            }


def measure_build_time_fit_vs_direct(X, y, k,n_repeats=5):
    """This function measure the improvement in time when we use direct tree construction instead of
    KNeighborsClassifier.fit() : overhead + tree construction
    """
    # store times for each method
    fit_times = {
        'kd_fit': -1.0,  # KNeighborsClassifier.fit() kd_tree
        'ball_fit': -1.0,  # KNeighborsClassifier.fit() ball_tree
        'kd_direct': -1.0,  # KDTree() directly
        'ball_direct': -1.0  # BallTree() directly
    }
    # --- KNeighborsClassifier.fit() ---
    for algo, key in [('kd_tree', 'kd_fit'), ('ball_tree', 'ball_fit')]:
        t_list = []
        for _ in range(n_repeats):
            knn = KNeighborsClassifier(n_neighbors=k, algorithm=algo)
            t0 = time.time()
            knn.fit(X, y)
            t_list.append(time.time() - t0)
        fit_times[key]= float(np.mean(t_list))

    # --- Direct tree construction ---
    for TreeClass, key in [(KDTree, 'kd_direct'), (BallTree, 'ball_direct')]:
        t_list = []
        for _ in range(n_repeats):
            t0 = time.time()
            tree = TreeClass(X, leaf_size=30)
            t_list.append(time.time() - t0)
        fit_times[key] = float(np.mean(t_list))

    return fit_times


def bestK(Xtrain, ytrain, minRange, maxRange, algo='brute', cv=5, Xvalid=None, yvalid=None):
    """Finds the best K that gives the smallest classification error"""
    err_valid = []
    # fix the folds explicitly (same splits for all algos)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    for k in range(minRange, maxRange+1):
        knn = KNeighborsClassifier(n_neighbors=k, algorithm=algo)

        #cross validation, cv different train/valid splits
        score = cross_val_score(knn, Xtrain, ytrain, cv=kf, scoring='accuracy')
        '''#Without cross validation
        knn.fit(Xtrain, ytrain)
        ypred = knn.predict(Xvalid)
        # risque de classification
        err_class = np.mean(yvalid != ypred)
        err_valid.append(err_class)'''

        err_valid.append(1 - score.mean()) #err = 1 - accuracy
        #print("Err de validation: " + str(err_class))

    '''print(f"[{algo}] Error curve: ")
    for i, err in enumerate(err_valid):
        print(f"  K={i + minRange}: {err:.6f}")'''
    # the best K with the smallest err
    best_K = np.argmin(err_valid) + minRange
    print(f"Best K {algo}: {best_K}, Error: {err_valid[best_K - minRange]:.4f}")

    '''plt.plot(range(minRange, maxRange + 1), err_valid, marker='o')
    plt.xlabel('K')
    plt.ylabel('Classification Error')
    plt.title(f'Classification Error vs K (algo={algo}, N= {Xtrain.shape[0]}, D= {Xtrain.shape[1]})')
    plt.axvline(x=best_K, color='red', linestyle='--', label=f'best k={best_K}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/bestK_{algo}.png", dpi=150)
    plt.show()'''

    return best_K

def bestLeafSize(Xtrain, Xtest, y,K,minRange, maxRange, algo, n_repeats=10):
    """For a dataset and Tree structure finds the best leaf size
    Mostly depends on Dimension and Number of samples"""
    leaf_sizes = range(minRange, maxRange + 1, 10)#10 by 10
    n_samples, n_features = Xtrain.shape
    n_test = len(Xtest)

    # warm-up data
    X_warmup = np.random.rand(2, Xtest.shape[1])  # same D, different points
    X_single = np.random.rand(1, Xtest.shape[1])  # single point, same D
    avg_query_times=[]
    time_per_points = []

    for ls in leaf_sizes:
        query_times = []
        knn = KNeighborsClassifier(n_neighbors=K, algorithm=algo, leaf_size=ls)
        knn.fit(Xtrain, y)
        knn.predict(X_warmup)  # warm up

        for _ in range(n_repeats):
            #query time

            start = time.time()
            y_pred = knn.predict(Xtest)
            query_t = time.time() - start
            query_times.append(query_t)

        avg_time = np.mean(query_times)
        avg_query_times.append(avg_time)
        time_per_points.append(avg_time / n_test)

    best_ind = np.argmin(avg_query_times)
    return list(leaf_sizes)[best_ind]




