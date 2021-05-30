"""
.. module:: gen_dbscan_dataset

DBSCAN
*************

:Description: Code to generate datasets approximately equal to the ones used in the DBSCAN paper


:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on: see version control

"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons

import unittest

__author__ = 'birkholz'

def gen_circle(center,radius,n,random_state=None):
    """Generate circular uniform distribution

    This function generates a uniform distribution of 2D points within a circular range

    Parameters:

    :param center: tuple with (x,y) of circle center
    :param radius: radius of circle
    :param n: number of points to generate

    :return: list of points (each point is a tuple)
    """
    circle_rng = np.random.default_rng() if random_state == None else np.random.default_rng(random_state)

    mag=[radius*np.sqrt(circle_rng.uniform(0,1)) for x in range(n)]
    phs=[np.pi*circle_rng.uniform(0,2) for x in range(n)]
    points=np.array([[center[0]+m*np.cos(p), center[1]+m*np.sin(p)] for m, p in zip(mag,phs)])

    return points

def gen_square(center,halflength,n,random_state=None):
    """Generate square uniform distribution

    This function generates a uniform distribution of 2D points within a square range

    Parameters:

    :param center: tuple with (x,y) of circle center
    :param halflength: half of the length of a side
    :param n: number of points to generate

    :return: list of points (each point is a tuple)
    """
    square_rng = np.random.default_rng() if random_state == None else np.random.default_rng(random_state)

    xoffset=[halflength*square_rng.uniform(-1,1) for x in range(n)]
    yoffset=[halflength*square_rng.uniform(-1,1) for x in range(n)]
    points=np.array([[center[0]+x, center[1]+y] for x, y in zip(xoffset,yoffset)])

    return points

def gen_rectangle(topleft,width,height,n,random_state=None):
    """Generate square uniform distribution

    This function generates a uniform distribution of 2D points within a square range

    Parameters:

    :param topleft: tuple with (x,y) of top left corner of rectangle
    :param width: width of rectangle
    :param height: height of rectangle
    :param n: number of points to generate

    :return: list of points (each point is a tuple)
    """
    rect_rng = np.random.default_rng() if random_state == None else np.random.default_rng(random_state)

    xoffset=[width*rect_rng.uniform(0,1) for x in range(n)]
    yoffset=[height*rect_rng.uniform(0,1) for x in range(n)]
    points=np.array([[topleft[0]+x, topleft[1]+y] for x, y in zip(xoffset,yoffset)])

    return points

def gen_shape(segments,radius,n,random_state=None):
    """Generate uniform distribution along a shape

    This function generates a uniform distribution along a set of line segments, using
    a radius parameter to control the width of the shape.

    Parameters:

    :param segments: list of line segments (each segment is a tuple of two (x,y) tuples)
    :param radius: radius of area around line segment to draw points, controls thickness of shape
    :n: number of points to generate

    :return: list of points (each point is a tuple)
    """
    # Start by picking random segment for each point and random length along segment
    # This isn't exactly right because shorter line segments will have a more dense
    # distribution than longer ones, but it's good enough. Additionally, it creates
    # interesting patterns of less-dense and more-dense areas within a cluster. If
    # there's time it can be rewritten to be sampled from a PDF built on line
    # segment length (or might be a good True/False parameter)
    shape_rng = np.random.default_rng() if random_state == None else np.random.default_rng(random_state)
    num_segments=len(segments)
    random_segments=shape_rng.integers(low=0,high=num_segments,size=n)
    random_lengths=[np.random.uniform(0,1) for x in range(n)]
    # Now, create a random set of points along the segments
    points=np.array([[segments[s][0][0]+l*(segments[s][1][0]-segments[s][0][0]),
                      segments[s][0][1]+l*(segments[s][1][1]-segments[s][0][1])]
                      for s, l in zip(random_segments,random_lengths)])

    # Use the random points we generated before as centers and add a random offset
    # within a circular radius in order to "thicken" the shape
    mag=[radius*np.sqrt(np.random.uniform(0,1)) for x in range(n)]
    phs=[np.pi*np.random.uniform(0,2) for x in range(n)]
    points=np.array([[c[0]+m*np.cos(p), c[1]+m*np.sin(p)] for m, p, c in zip(mag,phs,points)])

    return points

def gen_dbscan_dataset1(random_state=None):
    """Generate dataset similar to the first dataset in DBSCAN paper

    This function generates a dataset with four circles, each one uniform density

    Parameters:

    :return: list of points (each point is a tuple)
    """
    points=np.concatenate((gen_circle((500,500),315,400,random_state=random_state),
                           gen_circle((170,115),80,75,random_state=random_state),
                           gen_circle((828,128),85,75,random_state=random_state),
                           gen_circle((800,828),80,75,random_state=random_state)))

    return points

def gen_dbscan_dataset2(random_state=None):
    """Generate dataset similar to the second dataset in DBSCAN paper

    This function generates a dataset with four non-convex clusters

    Parameters:

    :return: list of points (each point is a tuple)
    """
    shape1 = [ ((692,626),(758,746)),
               ((758,746),(856,648)) ]
    width1 = 40
    shape2 = [ ((476,292),(674,436)),
               ((674,436),(908,358)),
               ((908,358),(930,528)) ]
    width2 = 40
    shape3 = [ ((350,560),(274,534)),
               ((274,534),(244,642)),
               ((244,642),(306,670)) ]
    width3 = 30

    shape4 = [ ((90,616),(114,720)),
               ((114,720),(306,852)),
               ((306,852),(446,758)),
               ((446,758),(522,616)),
               ((522,616),(522,528)),
               ((522,528),(316,298)),
               ((316,298),(300,232)),
               ((300,232),(380,166)),
               ((380,166),(564,194)),
               ((564,194),(660,294)) ]
    width4 = 40

    points=np.concatenate((gen_shape(shape1,width1,100,random_state=random_state),
                           gen_shape(shape2,width2,200,random_state=random_state),
                           gen_shape(shape3,width3,100,random_state=random_state),
                           gen_shape(shape4,width4,400,random_state=random_state)))

    return points

def gen_dbscan_dataset3(random_state=None):
    """Generate dataset similar to the third dataset in DBSCAN paper

    This function generates a dataset with two "shape-like" clusters
    and two circular ones, plus 10% of the points are background noise

    Parameters:

    :return: list of points (each point is a tuple)
    """
    shape1 = [ ((340,240),(645,250)),
               ((645,250),(715,340)),
               ((715,340),(810,375)),
               ((645,250),(692,190)),
               ((692,190),(787,155)),
               ((787,155),(848,143)) ]
    width1 = 30
    shape2 = [ ((781,254),(890,258)) ]
    width2 = 30

    points=np.concatenate((gen_shape(shape1,width1,300,random_state=random_state),
                           gen_shape(shape2,width2,100,random_state=random_state),
                           gen_circle((190,600),70,100,random_state=random_state),
                           gen_circle((680,700),125,200,random_state=random_state),
                           gen_square((500,500),500,70,random_state=random_state)))

    return points

def gen_dbscan_blobs(samples, blobs, std, random_state):
    """Generate dataset of a few blobs

    Parameters:
    :param samples: number of samples
    :param blobs: number of blobs

    :return: list of points (each point is a tuple)
    """
    X, y_true = make_blobs(n_samples=samples,
                           centers=blobs,
                           cluster_std=std,
                           random_state=random_state)

    return X

def gen_dbscan_moons(samples, noise=0.1,random_state=None):
    """Generate dataset of a few moons (non-convex)

    Parameters:
    :param samples: number of samples
    :param blobs: number of blobs

    :return: list of points (each point is a tuple)
    """
    X, y_true = make_moons(n_samples=samples, noise=noise)

    return X

def gen_gridbscan_synth10k(random_state=None):
    """Generate dataset similar to synthetic dataset of 10k points with
       10% noise

    Parameters:

    :return: list of points (each point is a tuple)
    """
    points=np.concatenate((gen_circle((275,175),150,7000,random_state=random_state),
                           gen_circle((600,100),50,1000,random_state=random_state),
                           gen_circle((600,250),50,1000,random_state=random_state),
                           gen_rectangle((0,0),700,350,1000,random_state=random_state)))

    return points


class TestGenDbscanDataset(unittest.TestCase):
    def test_gen_shape(self):
        print('\nRunning test_gen_shape:')

        segmentlist=[((25,50), (65,100)),((65,100),(30,85))]
        shape=gen_shape(segmentlist,1,1000)
        plt.figure(1)
        plt.scatter(shape[:,0], shape[:,1],s=2)
        plt.show()
        self.assertEqual(1,1)

    def test_gen_circle(self):
        print('\nRunning test_gen_circle:')

        circle=gen_circle((500,500),315,1000)
        plt.figure(1)
        plt.scatter(circle[:,0], circle[:,1],s=2)
        plt.show()
        self.assertEqual(1,1)

    def test_gen_square(self):
        print('\nRunning test_gen_square:')

        square=gen_square((500,500),50,10000)
        plt.figure(1)
        plt.scatter(square[:,0], square[:,1],s=2)
        plt.show()
        self.assertEqual(1,1)

    def test_gen_dbscan_dataset1(self):
        print('\nRunning test_gen_dbscan_dataset1:')

        dbscan_ds1=gen_dbscan_dataset1()
        plt.figure(1)
        plt.scatter(dbscan_ds1[:,0], dbscan_ds1[:,1],s=2)
        plt.show()

        self.assertEqual(1,1)

    def test_gen_dbscan_dataset2(self):
        print('\nRunning test_gen_dbscan_dataset2:')

        dbscan_ds2=gen_dbscan_dataset2()
        plt.figure(1)
        plt.scatter(dbscan_ds2[:,0], dbscan_ds2[:,1],s=2)
        plt.show()

        self.assertEqual(1,1)

    def test_gen_dbscan_dataset3(self):
        print('\nRunning test_gen_dbscan_dataset3:')

        dbscan_ds3=gen_dbscan_dataset3()
        plt.figure(1)
        plt.scatter(dbscan_ds3[:,0], dbscan_ds3[:,1],s=2)
        plt.show()

        self.assertEqual(1,1)

    def test_gen_gridbscan_synth10k(self):
        print('\nRunning test_gen_dbscan_dataset3:')

        synth10k=gen_gridbscan_synth10k()
        plt.figure(1)
        plt.scatter(synth10k[:,0], synth10k[:,1],s=2)
        plt.show()

        self.assertEqual(1,1)

if __name__ == '__main__':
    import unittest
    import logging

    import matplotlib.pyplot as plt

    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()
    # DEBUG is a good level for unit tests, but you can change to INFO if you want to shut it up
    # dbscan_logger=logging.get_logger('dbscan')
    # dbscan_logger.setLevel(logging.DEBUG)
    unittest.main()
