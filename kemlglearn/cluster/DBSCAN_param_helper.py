"""
.. module:: DBSCAN_param_helper

DBSCAN
*************

:Description: Utility to help find optimal parameters for DBSCAN based on the heuristic
              method proposed in Section 4.2 of the DBSCAN paper


:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on:

"""
import argparse

import pandas as pd
import numpy as np
from scipy.spatial import distance, distance_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons


from gen_dbscan_dataset import gen_dbscan_dataset1, gen_dbscan_dataset2, gen_dbscan_dataset3, \
                               gen_dbscan_blobs, gen_dbscan_moons
from DBSCAN import DBSCAN

__author__ = 'birkholz'

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

        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)

    # Show clusters (and noise)
    plt.title(f'Estimated number of clusters: {n_clusters_}')
    plt.show()

datagen_patterns = [ 'dbscan1', 'dbscan2', 'dbscan3', 'blobs', 'moons' ]

def gen_data(pattern):
    if pattern == 'dbscan1':
        data = gen_dbscan_dataset1()
    elif pattern == 'dbscan2':
        data = gen_dbscan_dataset2()
    elif pattern == 'dbscan3':
        data = gen_dbscan_dataset3()
    elif pattern == 'blobs':
        data = gen_dbscan_blobs(2000, 4, std=0.50, random_state=None)
    elif pattern == 'moons':
        data = gen_dbscan_moons(4000)
    return data

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", help="path to input data csv file")
    ap.add_argument("-g", "--generate", help="type of data to generate on the fly")
    args=vars(ap.parse_args())

    if args['generate']:
        if args['generate'] in datagen_patterns:
            X=gen_data(args['generate'])
        else:
            raise Exception('Try again. Could not understand requested pattern.')
    elif args['filename']:
        df=pd.read_csv(args['filename'],header=None)
        X=df.to_numpy()
    else:
        raise Exception('Need to specify filename or type of data to generate')

    distance_arr = distance_matrix(X,X)
    sorted_distance_arr=np.zeros_like(distance_arr)
    for i in range(distance_arr.shape[0]):
        sorted_distance_arr[i,:]=np.sort(distance_arr[i,:],)

    # Default to dims*2. This gives us the k-dist=4 suggested by the paper by default for 2D
    k=X.shape[1]*2
    # Or, dims+1 is another suggested choice
    #k=X.shape[1]+1

    stop=False
    while (not stop):
        sorted_kdist = np.sort(sorted_distance_arr[:,k])
        x = range(len(sorted_kdist))

        # Plot in descending order to match graph in paper
        plt.plot(x,sorted_kdist[::-1],'.')
        plt.title(str(k) + '-dist')
        plt.xlabel("Points sorted According to Distance of {}-dist Nearest Neighbor".format(str(k)))
        plt.ylabel("{}-Nearest Neighbor Distance".format(str(k)))
        plt.show()

        newk = input('Enter new k (leave blank to proceed with current value k=' + str(k) + '): ')
        if( newk == ''):
            stop = True
        else:
            try:
                newk = int(newk)
                if ((newk > 0) and (newk < len(sorted_distance_arr))):
                    k = newk
                else:
                    print('Invalid entry. Repeating last k...')
            except:
                print('Bad input. Repeating last k....')

    # Can override epsilon to avoid the prompt
    #eps=1
    if( 'eps' not in locals() ):
        eps = input('Enter epsilon: ')
        eps = float(eps)

    # Run DBSCAN and show results
    mydbscan = DBSCAN(eps=eps,min_samples=k).fit(X)
    plot_dbscan_results(X, mydbscan.labels_, mydbscan.core_sample_indices_)
