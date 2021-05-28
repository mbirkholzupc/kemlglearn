"""
.. module:: DBSCAN

DBSCAN
*************

:Description: DBSCAN algorithm


:Authors: birkholz

:License: BSD 3 clause (due to using API of DBSCAN from scikit-learn)

:Version:

:Created on: 1/4/2021

"""
import sys

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np

from collections import deque
import unittest
import time

__author__ = 'birkholz'

NOISE = -1
NO_CLUSTER = 0

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

    Attributes:
    core_sample_indices_: ndarray of shape(n_core_samples,)
      Indices of core samples (TODO: decide if this is sufficient or if it needs to tell us core/border/noise for each point for efficiency)
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
        # Set up neighbor query object based on requested algorithm
        # Refer to https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors
        nn=NearestNeighbors(radius=self.eps,algorithm=self.algorithm,
                            leaf_size=self.leaf_size,metric=self.metric,
                            p=self.p,metric_params=self.metric_params,
                            n_jobs=self.n_jobs)
        nn.fit(X)
        # Looking up the nearest neighbors on a point-by-point basis is very slow, so let's try
        # getting the full list here despite higher memory usage
        neighborlist = nn.radius_neighbors(X)

        # Default each label to 0 (NO_CLUSTER)
        self.labels_=np.zeros((len(X),), dtype=int)
        cur_cluster_label = NO_CLUSTER
        seeds=deque()
        self.core_sample_indices_ = []

        # Algorithm below is a near-verbatim implementation of the algorithm in
        # DBSCAN Revisited (Schubert, et al)
        # Added: core point indices
        for i,pt in enumerate(X):
            if self.labels_[i] != NO_CLUSTER:
                continue

            # neighborlist[0][x] contains distances, neighborlist[1][x] contains indices
            if len(neighborlist[0][i]) < self.min_samples:
                self.labels_[i] = NOISE
                continue

            # Increment cluster label and label this point
            cur_cluster_label += 1
            self.labels_[i] = cur_cluster_label

            # This is a core point
            self.core_sample_indices_.append(i)

            for newseed in neighborlist[1][i]:
                # Don't append current point
                if newseed != i:
                    seeds.append(newseed)

            while len(seeds) > 0:
                q = seeds.popleft()
                if self.labels_[q] == NOISE:
                    self.labels_[q] = cur_cluster_label
                if self.labels_[q] != NO_CLUSTER:
                    continue
                self.labels_[q] = cur_cluster_label
                if len(neighborlist[0][q]) < self.min_samples:
                    continue

                # If we get here, this is a core point
                self.core_sample_indices_.append(q)

                # Add new reachable points to seed list if we haven't checked them out yet
                for n in neighborlist[1][q]:
                    if self.labels_[n] == NOISE or self.labels_[n] == NO_CLUSTER:
                        seeds.append(n)

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Clusters the examples using DBSCAN algorithm and returns cluster labels for each sample
        :param X: data to cluster
        :param y: unused since this is unsupervised, but needed for API consistency
        :param sample_weight: unused (future: can make a single point count as more than 1)
        :return: predicted cluster labels
        """
        self.fit(X,sample_weight=None)
        return self.labels_
