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
from DBSCAN import DBSCAN, NOISE

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
    print(f'{gx} {gy} {gm}')
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
        if self.grid==None:
            # Behave like "normal" DBSCAN and don't divide into grid, i.e., just a single partition
            self.grid=[1]
        self.grid=np.array(self.grid)

        # Calculate the grid base we can use to flatten/blow up the grid
        # If dims are (w, x, y, z), then it is like this: (x*y*z, y*z, z, 1)
        cumprod=[np.prod(self.grid[0:x+1]) for x in range(len(self.grid))]
        self.gridbase=np.prod(self.grid)/cumprod

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
        print('Merge step')

        # Create cluster-equivalence list
        # TODO: Maybe this could be built up during DBSCAN iterations?
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

        print(cl_eq_list)

        # Create lists to track Cluster-ID, Core Flag and Potential Equivalence List
        cid         = np.zeros(len(X),int) # Group 0 is noise, remember
        core_flag   = np.full(len(X), False)
        pot_eq_list = np.empty(len(X),object)
        pot_eq_list[...]=[[] for _ in range(len(pot_eq_list))]

        with np.nditer(self.dummygrid,flags=['multi_index']) as it:
            for c in it:
                print('partition separator ' + str(it.multi_index))
                print(self.partitions[it.multi_index])
                print(self.partition_labels[it.multi_index])
                print(self.partition_corepts[it.multi_index])
                flat=self.to_flat(it.multi_index)
                gridcell=self.to_grid(flat)
                print(flat)
                print(gridcell)

                # Convert to set for fast lookup
                cur_corepts = set(self.partition_corepts[it.multi_index])
                # Save reference to labels for convenience
                cur_labels = self.partition_labels[it.multi_index]

                # Check each point and make sure we can go back from index in cluster to index
                # in full dataset
                for pt in self.partition_corepts[it.multi_index]:
                    print('local core: ' + str(pt) + '    global: ' + str(self.partitions[it.multi_index][pt]))

                for i,pt in enumerate(self.partition_labels[it.multi_index]):
                    print('partition: ' + str(i) + '(' + str(pt) + ')\tglobal: ' + str(self.partitions[it.multi_index][i]))

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
                                print('adding to pel: ' + str(clid))
                                pot_eq_list[glidx].append(clid)
                        else:
                            if core_flag[glidx]:
                                # If we know this is a core point already, merge clusters
                                merge_clusters(cl_eq_list, cid[glidx], clid)
                                if len(pot_eq_list[glidx]) != 0:
                                    print("WHATWHAT!")
                            else:
                                # Mark this point as a core point and set current cluster if it hasn't been set
                                core_flag[glidx]=True
                                cid[glidx] = clid
                                # Merge clusters in potential equivalence list
                                if len(pot_eq_list[glidx]) > 0:
                                    print(f'found non-empty potential equivalent list for {clid}. merging it...')
                                    for p in pot_eq_list[glidx]:
                                        print('p: ' + str(p))
                                        print('before')
                                        print(cl_eq_list)
                                        print(f'merging {cid[glidx]} and {p}...')
                                        merge_clusters(cl_eq_list, cid[glidx], p)
                                        print('after')
                                        print(cl_eq_list)
                                    # Now clear out list
                                    pot_eq_list[glidx] = []


                # print out the table of point/clusted/core flag/pot eq list
                for i in range(len(X)):
                    print(f'{i}: {cid[i]} {core_flag[i]} {pot_eq_list[i]}')

                if self.to_flat(it.multi_index) == 1:
                    #break
                    pass

        print('cl_eq_list')
        print(cl_eq_list)
        print('pre clusters')
        print(cid)
        # Now, if there are any non-empty potential equivalent lists, assign those points to a cluster
        for i,p in enumerate(pot_eq_list):
            if len(p) > 0:
                cid[i] = p[0]
        print('post clusters')
        print(cid)

        # Now we have everything we need, so let's set up return values
        self.labels_ = []
        self.core_sample_indices_ = []
        for i, cluster_core in enumerate(zip(cid,core_flag)):
            if cluster_core[0] == 0:
                self.labels_.append(NOISE)  # Translate noise from 0 to -1
            else:
                # Use the cluster group ID as the final label
                # TODO: This
                #self.labels_.append(cluster_core[0])
                self.labels_.append(cl_eq_list[cluster_core[0]])
            if cluster_core[1]:
                self.core_sample_indices_.append(i)
        print('cl_eq_list')
        print(cl_eq_list)

        self.labels_=np.array(self.labels_)
        self.core_sample_indices_=np.array(self.core_sample_indices_)

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

def dbscan_equivalent_results(labels1, corepts1, labels2, corepts2):
    # This is not the absolute most efficient implementation, but it
    # doesn't have to be since it's just a unit test.

    # Confirm results are identical. They should have all the same core points, but
    # it is possible for a border point to be grouped with a different cluster if it's
    # close enough to two different clusters.

    # Noise must be identical
    noise1=set(np.where(labels1==NOISE)[0])
    noise2=set(np.where(labels2==NOISE)[0])

    if noise1 != noise2:
        print('noise mismatch!')
        return False

    # Core samples must be identical (although they might belong to different clusters)
    if set(corepts1) != set(corepts2):
        print('core sample mismatch!')
        return False

    # Since the clusters may have different cluster IDs, we'll have to kind of guess which
    # clusters match.
    uniquelabels1 = np.unique(labels1)
    uniquelabels2 = np.unique(labels2)

    clusterdifferences = []

    for lbl1 in uniquelabels1:
        if lbl1 == -1:
            # Already checked noise, so continue with next cluster
            continue
        # Try to find most similar cluster
        set1=set(np.where(labels1==lbl1)[0])
        # Format of tuple: how many different samples there are, the list of different samples
        bestcluster = (None, None)
        for lbl2 in uniquelabels2:
            if lbl2 == -1:
                continue
            set2 = set(np.where(labels2==lbl2)[0])
            cluster_difference = set1 ^ set2
            mag_cluster_difference = len(cluster_difference)
            if bestcluster == (None,None):
                bestcluster = (mag_cluster_difference, cluster_difference)
            elif mag_cluster_difference < bestcluster[0]:
                bestcluster = (mag_cluster_difference, cluster_difference)

            # If they're identical, we can skip to the next iteration directly
            if mag_cluster_difference == 0:
                break

        if bestcluster[0] != None and bestcluster[0] > 0:
            clusterdifferences.append(bestcluster)

    if len(clusterdifferences) != 0:
        for difference in clusterdifferences:
            for pt in difference[1]:
                if pt in set(corepts1):
                    # If any core points are different, we're in trouble. Border
                    # points can be in different clusters though.
                    return False

    return True

# Helper function to plot DBSCAN results based on DBSCAN example in scikit-learn user guide
# If data has more than two dimensions, the first two will be used
def plot_dbscan_results(x, labels, core_sample_indices):
    core_samples_mask=np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1] # Black for noise

        class_member_mask = (labels == k)

        if k == 65539:
            xy = x[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

            xy = x[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=4)
        else:
            xy = x[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

        '''
        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)
        '''


    # Show clusters (and noise)
    plt.title(f'Estimated number of clusters: {n_clusters_}')
    plt.show()

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
                               random_state=2)
        X = X[:, ::-1]

        # Show points
        plt.figure(1)
        plt.scatter(X[:,0], X[:,1])
        plt.show()

        mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(6,2)).fit(X)

        print('GriDBSCAN')
        print(mygridbscan.labels_)
        print(mygridbscan.core_sample_indices_)

        print('DBSCAN')
        refdbscan=sklDBSCAN(eps=0.5,min_samples=4).fit(X)
        print(refdbscan.labels_)
        print(refdbscan.core_sample_indices_)

class TestGriDBSCANAuto(unittest.TestCase):
    def test_auto1(self):
        # 100 iterations generally seems to be enough to get an ambiguous border point
        for rs in range(1000):
            n_samples = 1000
            n_blobs = 4
            X, y_true = make_blobs(n_samples=n_samples,
                                   centers=n_blobs,
                                   cluster_std=0.60,
                                   random_state=rs)
            X = X[:, ::-1]

            '''
            # Shuffle the data to force (possibly) some border points to be different
            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            mydbscan=DBSCAN(eps=43,min_samples=4).fit(shuffleX)
            # Unshuffle the results so they can be compared with original indices
            unshuffled_labels=mydbscan.labels_[unshuffle]
            unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])
            mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(3,4)).fit(shuffleX)
            '''
            mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(3,4)).fit(X)

            refdbscan=sklDBSCAN(eps=0.5,min_samples=4).fit(X)
            print(mygridbscan.core_sample_indices_)
            print(refdbscan.core_sample_indices_)
            print(mygridbscan.labels_)
            print(refdbscan.labels_)

            uniquelabels1 = np.unique(mygridbscan.labels_)
            uniquelabels2 = np.unique(refdbscan.labels_)
            print(uniquelabels1)
            print(uniquelabels2)

            #plot_dbscan_results(X, mygridbscan.labels_, mygridbscan.core_sample_indices_)

            #plot_dbscan_results(X, refdbscan.labels_, refdbscan.core_sample_indices_)

            print(f'checking results for seed: {rs}')
            self.assertEqual(True,dbscan_equivalent_results(
                             #unshuffled_labels, unshuffled_corepts,
                             mygridbscan.labels_, mygridbscan.core_sample_indices_,
                             refdbscan.labels_, refdbscan.core_sample_indices_))


if __name__ == '__main__':
    import unittest
    import logging

    from sklearn.datasets import make_blobs, make_moons
    from scipy.spatial import distance
    from sklearn.metrics import pairwise
    import matplotlib.pyplot as plt
    from numpy.random import normal

    from sklearn.cluster import DBSCAN as sklDBSCAN

    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()
    # DEBUG is a good level for unit tests, but you can change to INFO if you want to shut it up
    # dbscan_logger=logging.get_logger('dbscan')
    # dbscan_logger.setLevel(logging.DEBUG)
    unittest.main()
