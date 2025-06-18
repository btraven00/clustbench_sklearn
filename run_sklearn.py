#!/usr/bin/env python

"""
Omnibenchmark-izes Markek Gagolewski's https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_sklearn.py

Takes the true number of clusters into account and outputs a 2D matrix with as many columns as ks tested,
being true number of clusters `k` and tested range `k plusminus 2`
"""

import argparse
import os, sys
import sklearn.cluster
import sklearn.mixture
import numpy as np
import warnings
import random
from prng import set_seed

VALID_METHODS = ['birch', 'kmeans', 'spectral', 'gm']

def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)

    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")

    return(data)

def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)

    ##data.reset_index(drop=True,inplace=True)

    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")

    return(data)

## this maps the ks to their true offset to the truth, e.g.:
# >>> generate_k_range(5)
# {'k-2': 3, 'k-1': 4, 'k': 5, 'k+1': 6, 'k+2': 7}
# >>> generate_k_range(1)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 2, 'k+2': 3}
# >>> generate_k_range(2)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 3, 'k+2': 4}
## k is the true k
def generate_k_range(k):
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    replace = lambda x: x if x >= 2 else 2 ## but we never run k < 2; those are replaced by a k=2 run (to not skip the calculation)
    Ks = list(map(replace, Ks))

    # ids = ['k-2', 'k-1', 'k', 'k+1', 'k+2']
    ids = list(range(0,5))
    assert(len(ids) == len(Ks))

    k_ids_dict = dict.fromkeys(ids, 0)
    for i in range(len(ids)):
        key = ids[i]

        k_ids_dict[key] = Ks[i]
    return(k_ids_dict)


def do_gm(X, Ks, seed=123):
    res = dict()
    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated
        with set_seed(seed):
            c = sklearn.mixture.GaussianMixture(n_components=K,
                n_init=100,
                # defaults: tol=1e-3, covariance_type="full", max_iter=100, reg_covar=1e-6
                random_state=seed
            )
            labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
            # print(K)
            # print(np.unique(labels_pred))
            # print('----')
            if len(np.unique(labels_pred)) != K: # some clusters might be empty, so not fullfiling the K requirement
                ## in that case, we report everything belongs to the K cluster
                res[K_id] = np.repeat(K, len(labels_pred))
            else:
                res[K_id] = labels_pred
    return np.array([res[key] for key in res.keys()]).T


## caution slow!
## FIXME(ben): for some reason, this uses a brute-force approach to find a
# working set of parameters for the spectral clustering algorithm.
# There are several problems with this approach:
# - It is computationally expensive, as it tries all possible combinations of parameters.
# - It does useless work since it overwrites the previous result on each iteration.
def do_spectral(X, Ks, seed=123):
    res = dict()
    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated
        for affinity in ["rbf", "laplacian", "poly", "sigmoid"]:
            for gamma in [0.25, 0.5, 1.0, 2.5, 5.0]:
                method = "sklearn_spectral_A%s_G%g"%(affinity, gamma)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with set_seed(seed):
                            c = sklearn.cluster.SpectralClustering(n_clusters=K,
                                affinity=affinity, gamma=gamma,
                                random_state=seed
                            )
                            labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
                            #print(np.bincount(labels_pred))
                            #print(len(labels_pred))
                            assert min(labels_pred) == 1
                            assert max(labels_pred) == K
                            assert labels_pred.shape[0] == X.shape[0]
                            assert len(np.unique(labels_pred)) == K
                            res[K_id] = labels_pred
                            # Once we find a valid configuration, we can break the inner loops
                            break
                except Exception as e:
                    # print(f"Error with SpectralClustering: {e}")
                    pass
            if K_id in res:
                break

    # Make sure all K_ids have an entry
    for item in Ks.keys():
        if item not in res:
            # If we couldn't find a valid clustering for this K,
            # create a default clustering where everything is in one cluster
            res[item] = np.ones(X.shape[0], dtype=int)

    return np.array([res[key] for key in res.keys()]).T

def do_kmeans(X, Ks, seed=123):
    res = dict()

    for K in Ks.keys(): res[K] = dict()

    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated
        with set_seed(seed):
            c = sklearn.cluster.KMeans(n_clusters=K,
                 # defaults: n_init=10, max_iter=300, tol=1e-4, init="k-means++"
                random_state=seed
            )
        labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
        res[K_id] = labels_pred
    return np.array([res[key] for key in res.keys()]).T

def do_birch(X, Ks, seed=123):
    res = dict()
    for K in Ks.keys(): res[K] = dict()

    # print(" >:", end="", flush=True)
    for branching_factor in [10, 50, 100]:
        for threshold in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]:
            for item in Ks.keys():
                K_id = item  ## just an unique identifier
                K = Ks[K_id] ## the tested k perhaps repeated
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # If threshold is too large, the number of subclusters
                        # found might be less than the requested one.
                        with set_seed(seed):
                            c = sklearn.cluster.Birch(n_clusters=K,
                                                    threshold=threshold,
                                                    branching_factor=branching_factor
                                                    )
                            labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
                            #print(np.bincount(labels_pred))
                            #print(len(labels_pred))

                            if labels_pred.max() == K:
                                res[K_id] = labels_pred
                except Exception as e:
                    # print(f"Error with Birch: {e}")
                    pass
            # print(".", end="", flush=True)
        # print(":", end="", flush=True)
    # print("<", end="", flush=True)
    # Make sure all K_ids have an entry
    for item in Ks.keys():
        if item not in res:
            # If we couldn't find a valid clustering for this K,
            # create a default clustering where everything is in one cluster
            res[item] = np.ones(X.shape[0], dtype=int)

    arr = np.array([res[key] for key in res.keys()]).T
    return arr


def main():
    parser = argparse.ArgumentParser(description='clustbench sklearn runner')

    parser.add_argument('--data.matrix', type=str,
                        help='gz-compressed textfile containing the comma-separated data to be clustered.', required = True)
    parser.add_argument('--data.true_labels', type=str,
                        help='gz-compressed textfile with the true labels; used to select a range of ks.', required = True)
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store data files.')
    parser.add_argument('--name', type=str, help='name of the dataset', default='clustbench')
    parser.add_argument('--method', type=str,
                        help='sklearn method',
                        required = True)
    parser.add_argument('--seed', type=int,
                        help='random seed for reproducibility',
                        required = False, default=123)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.method not in VALID_METHODS:
        raise ValueError(f"Invalid method `{args.method}`")

    truth = load_labels(getattr(args, 'data.true_labels'))
    k = int(max(truth)) # true number of clusters
    Ks = generate_k_range(k)

    data = getattr(args, 'data.matrix')
    dataset = load_dataset(data)
    if args.method == 'birch':
        curr = do_birch(X=dataset, Ks=Ks, seed=args.seed)
    elif args.method == 'kmeans':
        curr = do_kmeans(X=dataset, Ks=Ks, seed=args.seed)
    elif args.method == 'spectral':
        curr = do_spectral(X=dataset, Ks=Ks, seed=args.seed)
    elif args.method == 'gm':
        curr = do_gm(X=dataset, Ks=Ks, seed=args.seed)
    elif args.method in VALID_METHODS:
        raise ValueError('Valid method, but not implemented')
    else:
        raise ValueError(f"Invalid method `{args.method}`")

    name = args.name

    header=['k=%s'%s for s in Ks.values()]

    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(os.path.join(args.output_dir, f"{name}_ks_range.labels.gz"),
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header))

if __name__ == "__main__":
    main()
