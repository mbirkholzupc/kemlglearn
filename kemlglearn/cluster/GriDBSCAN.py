"""
.. module:: GriDBSCAN

GriDBSCAN
*************

:Description: GriDBSCAN algorithm based on 2008 paper by Shaaban Mahran and Khaled Mahar

:Authors: birkholz

:License: BSD 3 clause (due to using API of DBSCAN from scikit-learn)

:Version:

:Created on:

"""
# System/standard library imports
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import choice
import numpy as np

import time
import unittest

# Own imports

# Note: This is our own custom version of DBSCAN, not the sklearn version, but it should be drop-in compatible
from DBSCAN import DBSCAN

__author__ = 'birkholz'

class GriDBSCAN(BaseEstimator, ClusterMixin):
    """GriDBSCAN Implementation

    Partitions a dataset using the GriDBSCAN algorithm. Note that the API follows the scikit-learn
    conventions.

    Parameters:

    eps: epsilon, which defines the radius of the neighborhood of a point. default=0.5
    min_samples: minimum number of samples that must be in the neighborhood of a point for it to be considered a core point. default=5
    metric: distance metric. default='euclidean'
    metric_params: unused, just for consistency with scikit-learn API. default=None
    algorithm: nearest-neighbor search algorithm. one of: {'auto', 'ball_tree', 'kd_tree', 'brute'}. default='auto'
    leaf_size: leaf size for ball_tree or kd_tree. default=30
    p: power of the Minkowski metric for distance calculation. If None, then p=2 (Euclidean). default=None
    n_jobs: number of jobs to run in parallel. ignored for now. default=None, which forces single-CPU operation

    Attributes:
    core_sample_indices_: ndarray of shape(n_core_samples,)
      Indices of core samples (TODO: decide if this is sufficient or if it needs to tell us core/border/noise for each point for efficiency)
    """
    #TODO: Decide if we should pass through all parameters to DBSCAN or only support a reduced set

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None,
                 algorithm='auto', leaf_size=30, p=None, n_jobs=None, grid=None):
        self.eps=eps
        self.min_samples=min_samples
        self.metric=metric
        self.metric_params=metric_params
        self.algorithm=algorithm
        self.leaf_size=leaf_size
        self.p=p
        self.n_jobs=n_jobs
        self.grid=grid

    def fit(self, X, y=None, sample_weight=None):
        """
        Clusters the examples using DBSCAN algorithm and returns DBSCAN object
        :param X: data to cluster
        :param y: unused since this is unsupervised, but needed for API consistency
        :param sample_weight: unused (future: can make a single point count as more than 1)
        :return: DBSCAN object after clustering
        """

        # Do GriDBSCAN here
        print('Would be doing GriDBSCAN...')

        # Step 1: Build grid
        if self.grid==None:
            # Behave like "normal" DBSCAN and don't divide into grid, i.e., one square in all dims
            # Need to defer this step to the "fit" call since we don't have the data dimensions yet
            # in the constructor
            self.grid=[1 for x in range(len(X[0]))]
        self.grid=np.array(self.grid)

        # Note: Could make this O(n) instead of #attributes*O(n), but supposedly O(n) time is fairly
        #       negligible compared to the O(n^2) the DBCAN portion takes
        #       My first attempt at only traversing the array once was foiled by python's implementation
        #       and was about 10x as slow but maybe there's a way to get the best of both worlds with
        #       a lambda?
        # TODO: Measure and find out if it matters
        '''
        start_time=time.time()
        mins = [x for x in X[0]]
        maxs = [x for x in X[0]]
        for row in X:
            for i,val in enumerate(row):
                if val < mins[i]:
                    mins[i]=val
                elif val > maxs[i]:
                    maxs[i]=val
        end_time=time.time()
        print(X[0])
        print(mins)
        print(maxs)
        print('time: ' + str(end_time-start_time))
        '''

        # Less code, but takes longer
        #start_time=time.time()
        #mins=[min(X[:,i]) for i in range(len(X[0]))]
        #maxs=[max(X[:,i]) for i in range(len(X[0]))]
        #end_time=time.time()

        mins=[]
        maxs=[]
        for i in range(len(X[0])):
            mins.append(min(X[:,i]))
            maxs.append(max(X[:,i]))
        print('mins: ' + str(mins))
        print('maxs: ' + str(maxs))
        mins=np.array(mins)
        maxs=np.array(maxs)

        # Calculate and validate grid spacings
        dimrange=[themax-themin for themax, themin in zip(maxs,mins)]
        dimrange=np.array(dimrange)
        print(dimrange)
        gridspacing=dimrange/self.grid
        print(gridspacing)

        if any(space < 2*self.eps for space in gridspacing):
            # TODO: Tell them which dimension and specific numbers so they can fix it
            raise Exception("Error: Invalid grid spacing detected. Reduce number of partitions in at least one dimension.")

        # Step 2: Group points into partitions
        # Create a grid in D-dimensional space with a numpy array that can hold an object per cell.
        # We will put a reference to a regular python list in each cell since it is an efficient
        # structure to use for a growing list. Each list will hold the indices of all inner and outer
        # points for that cell. There does not seem to be a need to distinguish between inner/outer
        # points in the implementiation, just in the proof of correctness.
        self.partitions = np.empty(self.grid,dtype='object')

        # This is kind of dumb, but if we set up an empty object to iterate over, we can iterate over
        # cells easily without having to know the dimensions. The nditer's multi_index will give us the
        # cell index when needed. It also saves us from having to jump through lots of hoops regarding
        # modifying the thing we're iterating over and allowing references
        self.dummygrid = np.zeros(self.grid)

        with np.nditer(self.dummygrid, flags=['multi_index']) as it:
            for c in it:
                # Start out with an empty list of points in each cell
                self.partitions[it.multi_index]=list()

                # Calculate grid cell boundaries
                llimit=mins+gridspacing*it.multi_index-self.eps
                ulimit=mins+gridspacing*it.multi_index+gridspacing+self.eps
                for i,pt in enumerate(X):
                    # Going to make the epsilon-boundary inclusive on both upper/lower bounds
                    # Could potentially result in a few extra CPU cycles here and there, but
                    # should be minimal impact
                    if (llimit <= pt).all() and (pt <= ulimit).all():
                        self.partitions[it.multi_index].append(i)

        """
        # TODO: See if I should revive this code to speed things up. The other way is C*O(n), but
        #       this way could potentially be O(n) * <number much smaller than C> since I think most points won't
        #       be outside points in general (at least when grid dimensions are quite a bit larger than 2*eps
        print('partitions')
        for idx,pt in enumerate(X):
            # This calculates inner points
            # It was a lot easier to read like this:
            #partition = tuple([int((a-b)/c) for a,b,c in zip(pt, mins, gridspacing)])
            # But, there's a corner case where the max point in each dimension will get put in an out of bounds
            # grid space, so move it back into the grid if it's right on the edge (using d, e below)
            partition = tuple([int((a-b)/c) if a!=d else (e-1) for a,b,c,d,e in zip(pt, mins, gridspacing, maxs, self.grid)])
            try:
                self.inner_pts[partition].append(list(pt))
            except AttributeError:
                self.inner_pts[partition]=list(pt)

            # Now check if the point is close enough to any edges of the partition that it could be an outer
            # point in another partition. It is most efficient to check if it is guaranteed to only be an
            # inner point first since we can get that with simple math. After that, if it's near an edge
            # or multiple edges, there is more math to do. But, if we can just detect which dimensions
            # this condition applies to, we can easily add the point to all partitions where it's an outer
            # point.
            iimin = [m+p*s+self.eps for p,m,s in zip(partition,mins,gridspacing)]
            iimax = [m+(p+1)*s-self.eps for p,m,s in zip(partition,mins,gridspacing)]
            print('iimin: ' + str(iimin))
            print('iimax: ' + str(iimax))
            closelow = TODO

            print('partition: ' + str(partition))
            print(pt)
            print(mins)
            print(gridspacing)
            print(partition)
            if idx==2:
                break
        """
        
        '''
        for idx,pt in enumerate(X):
            # This calculates inner points
            partition = tuple([int((a-b)/c) for a,b,c in zip(pt, mins, gridspacing)])
            try:
                self.inner_pts[partition].append(list(pt))
            except KeyError:
                self.inner_pts[partition]=list(pt)


            print(pt)
            print(mins)
            print(gridspacing)
            print(partition)
            if idx==2:
                break
        '''
        print('partitions')
        print(self.partitions)
        with np.nditer(self.dummygrid,flags=['multi_index']) as it:
            for c in it:
                print(str(it.multi_index) + ': ' + str(len(self.partitions[it.multi_index])))

        # Step 3: Run DBSCAN on each partition
        # Note: This step would be the highest-priority function to parallelize as it takes O(n/C)^2 time.
        #       However, the paper compares only single-CPU/single-thread performance, so that's the
        #       benchmark we'll stick with.

        # Create D-dimensional grids to hold results (labels, core points) of DBSCAN for each partition
        self.partition_labels = np.empty(self.grid,dtype='object')
        self.partition_corepts = np.empty(self.grid,dtype='object')

        with np.nditer(self.dummygrid,flags=['multi_index']) as it:
            for c in it:
                # Run DBSCAN only on the points in this partition
                partX=X[self.partitions[it.multi_index]]
                if len(partX) > 0:
                    dbscan=DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric,
                                  metric_params=self.metric_params, algorithm=self.algorithm,
                                  leaf_size=self.leaf_size, p=self.p, n_jobs=self.n_jobs).fit(partX)
                    self.partition_labels[it.multi_index]=dbscan.labels_
                    self.partition_corepts[it.multi_index]=dbscan.core_sample_indices_
                else:
                    self.partition_labels[it.multi_index]=np.array([])
                    self.partition_corepts[it.multi_index]=np.array([])
                print('Ran DBSCAN for ' + str(it.multi_index))
                print(self.partition_labels[it.multi_index])
                print(self.partition_corepts[it.multi_index])

        # Step 4: Merge clusters

        


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

class TestGriDBSCAN(unittest.TestCase):
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

        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10).fit(X)

        print('Done with test_two.')
        self.assertEqual(1,1)

    def test_minmax(self):
        print('\nRunning test_minmax:')
        n_samples = 10000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]
        print(X)

        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10).fit(X)
        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10,grid=(2,2)).fit(X)
        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10,grid=(6,2)).fit(X)

    def test_innerouter(self):
        print('\nRunning test_innerouter:')
        X = np.array( [ [  8, 8],
                        [2.1, 4],
                        [ -4, 3],
                        [  0, 0],
                        [  3,-2],
                        [  4, 1],
                        [  0, 4.1] ] )
        print(X)

        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10,grid=(4,5)).fit(X)

    def test_blobs(self):
        print('\nRunning test_blobs:')
        n_samples = 1000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]

        mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(6,2)).fit(X)

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
