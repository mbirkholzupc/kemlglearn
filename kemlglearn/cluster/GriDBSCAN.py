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
from kemlglearn.cluster.DBSCAN import DBSCAN, NOISE

__author__ = 'birkholz'

# Reserve space for 65535 clusters per partition (unlikely we'd ever get that high)
# Keep in mind "noise" is encoded as -1 (hence why 65535, not 65536)
# Also, remember this isn't a uint32
PART_MULT=(1<<16)
PART_MASK=(0xffff0000)
CLST_MASK=(0xffff)

def to_clid(partition,cluster):
    """
    Generate a unique cluster ID based on partition number and cluster within partition

    :param partition: a number representing the partition (flattened, not a tuple)
    :param cluster: an int representing the cluster within the partition

    :return: a unique cluster ID
    """
    # Add one to move noise from -1 to 0
    clid = int((partition*PART_MULT)+cluster+1)
    return clid

def is_noise(clid):
    """
    Check if a cluster ID is noise

    :param clid: an int representing the cluster ID

    :return: True if noise, False if "real" cluster
    """
    return True if ((clid & CLST_MASK)==0) else False

def merge_clusters(cl_eq_list, cx, cy):
    # Note: Different from paper but I think paper has an error since the merge they propose
    #       fails if gx > gy

    # Should never be called with noise because noise can't be a core point
    assert(cx!=0)
    assert(cy!=0)

    gx=cl_eq_list[cx]
    gy=cl_eq_list[cy]
    gm=min(gx,gy)
    for k in cl_eq_list:
        if cl_eq_list[k] == gx or cl_eq_list[k] == gy:
            cl_eq_list[k] = gm

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
      Indices of core samples
    """
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
        if self.grid==None:
            # Behave like "normal" DBSCAN and don't divide into grid, i.e., just a single partition
            self.grid=[1]
        self.grid=np.array(self.grid)

        # Calculate the grid base we can use to flatten/blow up the grid
        # If dims are (w, x, y, z), then it is like this: (x*y*z, y*z, z, 1)
        cumprod=[np.prod(self.grid[0:x+1]) for x in range(len(self.grid))]
        self.gridbase=np.prod(self.grid)/cumprod
        self.dbscan_total_time_ = 0

    def _step1_build_grid(self, X):
        """
        Build the grid and validate the spacing
        Note: Could make this O(n) instead of #attributes*O(n), but supposedly O(n) time is fairly
              negligible compared to the O(n^2) the DBCAN portion takes
              My first attempt at only traversing the array once was foiled by python's implementation
              and was about 10x as slow but maybe there's a way to get the best of both worlds with
              a lambda?
        """
        mins=[]
        maxs=[]
        for i in range(len(X[0])):
            mins.append(min(X[:,i]))
            maxs.append(max(X[:,i]))
        mins=np.array(mins)
        maxs=np.array(maxs)

        # Less code, but takes longer unfortunately, so going with what I did above
        #start_time=time.time()
        #mins=[min(X[:,i]) for i in range(len(X[0]))]
        #maxs=[max(X[:,i]) for i in range(len(X[0]))]
        #end_time=time.time()

        # Calculate and validate grid spacings
        dimrange=[themax-themin for themax, themin in zip(maxs,mins)]
        dimrange=np.array(dimrange)
        gridspacing=dimrange/self.grid

        if any(space < 2*self.eps for space in gridspacing):
            # TODO: Tell them which dimension and specific numbers so they can fix it
            raise Exception("Error: Invalid grid spacing detected. Reduce number of partitions in at least one dimension.")

        return mins, maxs, gridspacing

    def _step2_partition_points(self,X,mins,gridspacing):
        """
        Partition points into grid cells
        """
        # Create a grid in D-dimensional space with a numpy array that can hold an object per cell.
        # We will put a reference to a regular python list in each cell since it is an efficient
        # structure to use for a growing list. Each list will hold the indices of all inner and outer
        # points for that cell. There does not seem to be a need to distinguish between inner/outer
        # points in the implementiation, just in the proof of correctness.
        self.partitions = np.empty(self.grid,dtype='object')

        # This is kind of dumb, but if we set up an empty object to iterate over, we can iterate over
        # cells easily without having to know the dimensions explicitly (just make it the same shape as
        # the grid). Even though we're ultimately going to have to flatten the data, the multi_index is
        # useful to use python's efficient vectorized math functions. It also saves us from having to jump
        # through lots of hoops regarding modifying the thing we're iterating over and allowing references
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

        return

    def _step3_run_dbscan(self,X):
        """
        Run DBSCAN on each grid cell
        No inputs/outputs since everything belongs to the class
        Note: This step would be the highest-priority function to parallelize as it takes O((n/C)^2) time
              and each DBSCAN can be run totally independently of the others. However, the paper compares
              only single-CPU/single-thread performance, so that's the benchmark we'll stick with.
        """
        # Create D-dimensional grids to hold results (labels, core points) of DBSCAN for each partition
        self.partition_labels = np.empty(self.grid,dtype='object')
        self.partition_corepts = np.empty(self.grid,dtype='object')

        with np.nditer(self.dummygrid,flags=['multi_index']) as it:
            for c in it:
                # Run DBSCAN only on the points in this partition
                partX=X[self.partitions[it.multi_index]]
                if len(partX) > 0:
                    start_time=time.time()
                    dbscan=DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric,
                                  metric_params=self.metric_params, algorithm=self.algorithm,
                                  leaf_size=self.leaf_size, p=self.p, n_jobs=self.n_jobs).fit(partX)
                    end_time=time.time()
                    self.dbscan_total_time_ += (end_time-start_time)
                    self.partition_labels[it.multi_index]=dbscan.labels_
                    self.partition_corepts[it.multi_index]=dbscan.core_sample_indices_
                else:
                    self.partition_labels[it.multi_index]=np.array([])
                    self.partition_corepts[it.multi_index]=np.array([])

        return

    def _step4_merge_clusters(self,X):
        """
        Merge the clusters according to GriDBSCAN paper's algorithm
        All inputs/outputs passed through class
        """
        # Create cluster-equivalence list
        # TODO: Maybe this could be built up during DBSCAN iterations? Doesn't matter much since this is so
        #       short compared to running DBSCAN
        cl_eq_list = {}
        with np.nditer(self.dummygrid,flags=['multi_index']) as it:
            for c in it:
                clusters_in_partition = np.unique(self.partition_labels[it.multi_index])
                flatpartition = self.to_flat(it.multi_index)
                for cip in clusters_in_partition:
                    clid = to_clid(flatpartition, cip)
                    if not is_noise(clid):
                        # Key is cluster ID, value is group ID. They start off the same.
                        cl_eq_list[clid]=clid

        # Create lists to track Cluster-ID, Core Flag and Potential Equivalence List
        cid         = np.zeros(len(X),int) # Group 0 is noise, remember
        core_flag   = np.full(len(X), False)
        pot_eq_list = np.empty(len(X),object)
        pot_eq_list[...]=[[] for _ in range(len(pot_eq_list))]

        with np.nditer(self.dummygrid,flags=['multi_index']) as it:
            for c in it:
                flat=self.to_flat(it.multi_index)
                gridcell=self.to_grid(flat)

                # Convert to set for fast lookup
                cur_corepts = set(self.partition_corepts[it.multi_index])
                # Save reference to labels for convenience
                cur_labels = self.partition_labels[it.multi_index]

                # For each point in this cluster, get it indexed in the Cl-ID, Core Flag and Potential
                # Equivalence List and try to merge equivalent clusters
                for pt,lbl in enumerate(self.partition_labels[it.multi_index]):
                    # Look up index of this point in the full database and calculate Cluster ID
                    glidx=self.partitions[it.multi_index][pt]
                    clid=to_clid(flat,lbl)

                    if cur_labels[pt] != NOISE:
                        if pt not in cur_corepts:
                            if core_flag[glidx]:
                                merge_clusters(cl_eq_list, cid[glidx], clid)
                            else:
                                # Add to potential equivalence list
                                clid=to_clid(flat,lbl)
                                pot_eq_list[glidx].append(clid)
                        else:
                            if core_flag[glidx]:
                                # If we know this is a core point already, merge clusters
                                merge_clusters(cl_eq_list, cid[glidx], clid)
                                if len(pot_eq_list[glidx]) != 0:
                                    # Should never see this
                                    print("WHATWHAT!")
                            else:
                                # Mark this point as a core point and set current cluster if it hasn't been set
                                core_flag[glidx]=True
                                cid[glidx] = clid
                                # Merge clusters in potential equivalence list
                                if len(pot_eq_list[glidx]) > 0:
                                    for p in pot_eq_list[glidx]:
                                        merge_clusters(cl_eq_list, cid[glidx], p)
                                    # Now clear out list
                                    pot_eq_list[glidx] = []

        # Now, if there are any non-empty potential equivalent lists, assign those points to a cluster
        for i,p in enumerate(pot_eq_list):
            if len(p) > 0:
                cid[i] = p[0]

        # Now we have everything we need, so let's set up return values
        self.labels_ = []
        self.core_sample_indices_ = []
        for i, cluster_core in enumerate(zip(cid,core_flag)):
            if cluster_core[0] == 0:
                self.labels_.append(NOISE)  # Translate noise from 0 to -1
            else:
                # Use the cluster group ID as the final label
                self.labels_.append(cl_eq_list[cluster_core[0]])
            if cluster_core[1]:
                self.core_sample_indices_.append(i)

        self.labels_=np.array(self.labels_)
        self.core_sample_indices_=np.array(self.core_sample_indices_)

        return

    def fit(self, X, y=None, sample_weight=None):
        """
        Clusters the examples using DBSCAN algorithm and returns DBSCAN object
        :param X: data to cluster
        :param y: unused since this is unsupervised, but needed for API consistency
        :param sample_weight: unused (future: can make a single point count as more than 1)
        :return: DBSCAN object after clustering
        """
        # Do GriDBSCAN here
        # Step 1: Build grid
        mins, maxs, grid_spacing = self._step1_build_grid(X)

        # Step 2: Group points into partitions
        self._step2_partition_points(X,mins,grid_spacing)

        # Step 3: Run DBSCAN on each partition
        self._step3_run_dbscan(X)

        # Step 4: Merge clusters
        self._step4_merge_clusters(X)

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

    def to_flat(self,cell):
        return np.dot(self.gridbase,cell)

    def to_grid(self,flat):
        buildcell=[]
        remaining=flat
        for i,x in enumerate(self.gridbase):
            buildcell.append(int(remaining/self.gridbase[i]))
            remaining = remaining % self.gridbase[i]
        return tuple(buildcell)

