"""
.. module:: DBSCAN

DBSCAN
*************

:Description: DBSCAN algorithm


:Authors: birkholz

:License: BSD 3 clause (due to using API of DBSACN from scikit-learn)

:Version:

:Created on: 1/4/2021

"""
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import choice
import numpy as np

import unittest

__author__ = 'birkholz'

class DBSCAN(BaseEstimator, ClusterMixin):
    """DBSCAN Implementation

    Partitions a dataset using the DBSCAN algorithm. Note that the API is taken directly from
    scikit-learn in order to be a drop-in replacement, but the implementation is completely unique.
    As such, not all of the parameters may be supported in this implementation.

    Parameters:

    eps: epsilon, which defines the radius of the neighborhood of a point. default=0.5
    min_samples: minimum number of samples that must be in the neighborhood of a point for it to be considered a core point. default=5
    metric: distance metric. default='euclidean'
    metric_params: unused, just for consistency with scikit-learn API. default=None
    algorithm: nearest-neighbor search algorithm. one of: {'auto', 'ball_tree', 'kd_tree', 'brute'}. default='auto'
    leaf_size: leaf size for ball_tree or kd_tree. default=30
    p: power of the Minkowski metric for distance calculation. If None, then p=2 (Euclidean). default=None
    n_jobs: number of jobs to run in parallel. ignored for now. default=None
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None,
                 algorithm='auto', leaf_size=30, p=None, n_jobs=None):
        self.eps=eps
        self.min_samples=min_samples
        self.metric=metric
        self.metric_params=metric_params
        self.algorithm=algorithm
        self.leaf_size=leaf_size
        self.p=p
        self.n_jobs=n_jobs

    def fit(self, X, y=None, sample_weight=None):
        """
        Clusters the examples using DBSCAN algorithm and returns DBSCAN object
        :param X: data to cluster
        :param y: unused since this is unsupervised, but needed for API consistency
        :param sample_weight: unused (future: can make a single point count as more than 1)
        :return: DBSCAN object after clustering
        """

        # Do DBSCAN here
        print('Would be doing DBSCAN...')

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Clusters the examples using DBSCAN algorithm and returns cluster labels for each sample
        :param X: data to cluster
        :param y: unused since this is unsupervised, but needed for API consistency
        :param sample_weight: unused (future: can make a single point count as more than 1)
        :return: predicted cluster labels
        """
        self.labels_ = self.fit(X,sample_weight=None)
        return self.labels_

class TestDBSCAN(unittest.TestCase):
    def test_one(self):
        print('\nRunning test_one:')
        n_samples = 4000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]

        plt.figure(1)
        plt.scatter(X[:,0], X[:,1])
        plt.show()

        X, y_true = make_moons(n_samples=n_samples, noise=0.1)
        plt.figure(1)
        plt.scatter(X[:,0], X[:,1], s=100)
        plt.show()
        print('Done with test_one.')
        self.assertEqual(1,1)

    def test_two(self):
        print('\nRunning test_two:')
        n_samples = 4000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]

        mydbscan=DBSCAN(eps=0.2,min_samples=10).fit(X)

        print('Done with test_two.')
        self.assertEqual(1,1)

if __name__ == '__main__':
    import unittest
    import logging

    from sklearn.datasets import make_blobs, make_moons
    from scipy.spatial import distance
    from sklearn.metrics import pairwise
    import matplotlib.pyplot as plt
    from numpy.random import normal

    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()
    # DEBUG is a good level for unit tests, but you can change to INFO if you want to shut it up
    # dbscan_logger=logging.get_logger('dbscan')
    # dbscan_logger.setLevel(logging.DEBUG)
    unittest.main()
