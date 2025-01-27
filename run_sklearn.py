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


def do_gm(X, Ks):
    res = dict()

    for K in Ks:
        c = sklearn.mixture.GaussianMixture(n_components=K,
            n_init=100,
            # defaults: tol=1e-3, covariance_type="full", max_iter=100, reg_covar=1e-6
            random_state=123
        )
        labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
        # print(K)
        # print(np.unique(labels_pred))
        # print('----')
        if len(np.unique(labels_pred)) != K: # some clusters might be empty, so not fullfiling the K requirement
            ## in that case, we report everything belongs to the K cluster
            res[K] =  np.repeat(K, len(labels_pred))
        res[K] = labels_pred
    return np.array([res[key] for key in res.keys()]).T


## caution slow!
def do_spectral(X, Ks):
    res = dict()

    for K in Ks:
        for affinity in ["rbf", "laplacian", "poly", "sigmoid"]:
            for gamma in [0.25, 0.5, 1.0, 2.5, 5.0]:
                method = "sklearn_spectral_A%s_G%g"%(affinity, gamma)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        c = sklearn.cluster.SpectralClustering(n_clusters=K,
                            affinity=affinity, gamma=gamma,
                            random_state=123
                        )
                        labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
                        #print(np.bincount(labels_pred))
                        #print(len(labels_pred))
                        assert min(labels_pred) == 1
                        assert max(labels_pred) == K
                        assert labels_pred.shape[0] == X.shape[0]
                        assert len(np.unique(labels_pred)) == K
                        res[K] = labels_pred
                except:
                    pass
    return np.array([res[key] for key in res.keys()]).T

def do_kmeans(X, Ks):
    res = dict()
    for K in Ks: res[K] = dict()

    for K in Ks:
        c = sklearn.cluster.KMeans(n_clusters=K,
             # defaults: n_init=10, max_iter=300, tol=1e-4, init="k-means++"
            random_state=123
        )
        labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
        res[K] = labels_pred
    return np.array([res[key] for key in res.keys()]).T

def do_birch(X, Ks):
    res = dict()
    for K in Ks: res[K] = dict()

    # print(" >:", end="", flush=True)
    for branching_factor in [10, 50, 100]:
        for threshold in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]:
            for K in Ks:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # If threshold is too large, the number of subclusters
                        # found might be less than the requested one.
                    c = sklearn.cluster.Birch(n_clusters=K,
                                              threshold=threshold,
                                              branching_factor=branching_factor
                                              )
                    labels_pred = c.fit_predict(X)+1 # 0-based -> 1-based
                    #print(np.bincount(labels_pred))
                    #print(len(labels_pred))
                except:
                    pass
                if labels_pred.max() == K:
                    res[K] = labels_pred
            # print(".", end="", flush=True)
        # print(":", end="", flush=True)
    # print("<", end="", flush=True)
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

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.method not in VALID_METHODS:
        raise ValueError(f"Invalid method `{args.method}`")

    truth = load_labels(getattr(args, 'data.true_labels'))
    k = int(max(truth)) # true number of clusters
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    replace = lambda x: x if x >= 2 else 2 ## but we never run k < 2; those are replaced by a k=2 run (to not skip the calculation)
    Ks = list(map(replace, Ks))
    
    data = getattr(args, 'data.matrix')
    if args.method == 'birch':
        curr = do_birch(X= load_dataset(data), Ks = Ks)
    elif args.method == 'kmeans':
        curr = do_kmeans(X= load_dataset(data), Ks = Ks)
    elif args.method == 'spectral':
        curr = do_spectral(X = load_dataset(data), Ks = Ks)
    elif args.method == 'gm':
        curr = do_gm(X = load_dataset(data), Ks = Ks)
    elif args.method in VALID_METHODS:
        raise ValueError('Valid method, but not implemented')
    
    name = args.name

    header=['k=%s'%s for s in Ks]

    print(curr.shape)
    
    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(os.path.join(args.output_dir, f"{name}_ks_range.labels.gz"),
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header)) 

if __name__ == "__main__":
    main()
