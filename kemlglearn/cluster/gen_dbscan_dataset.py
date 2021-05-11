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

import unittest

__author__ = 'birkholz'

# Random number generator
the_rng = np.random.default_rng()

def gen_circle(center,radius,n):
    """Generate circular uniform distribution

    This function generates a uniform distribution of 2D points within a circular range

    Parameters:

    :param center: tuple with (x,y) of circle center
    :param radius: radius of circle
    :param n: number of points to generate

    :return: list of points (each point is a tuple)
    """
    mag=[radius*np.sqrt(np.random.uniform(0,1)) for x in range(n)]
    phs=[np.pi*np.random.uniform(0,2) for x in range(n)]
    points=np.array([[center[0]+m*np.cos(p), center[1]+m*np.sin(p)] for m, p in zip(mag,phs)])

    return points

def gen_square(center,halflength,n):
    """Generate square uniform distribution

    This function generates a uniform distribution of 2D points within a square range

    Parameters:

    :param center: tuple with (x,y) of circle center
    :param halflength: half of the length of a side
    :param n: number of points to generate

    :return: list of points (each point is a tuple)
    """
    xoffset=[halflength*np.random.uniform(-1,1) for x in range(n)]
    yoffset=[halflength*np.random.uniform(-1,1) for x in range(n)]
    points=np.array([[center[0]+x, center[1]+y] for x, y in zip(xoffset,yoffset)])

    return points

def gen_shape(segments,radius,n):
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
    # distribution than longer ones, but it's good enough. If there's time it can
    # be rewritten to be sampled from a PDF built on line segment length
    num_segments=len(segments)
    random_segments=the_rng.integers(low=0,high=num_segments,size=n)
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

def gen_dbscan_dataset1():
    """Generate dataset similar to the first dataset in DBSCAN paper

    This function generates a dataset with four circles, each one uniform density

    Parameters:

    :return: list of points (each point is a tuple)
    """
    points=np.concatenate((gen_circle((500,500),315,400),
                           gen_circle((170,885),80,75),
                           gen_circle((828,128),85,75),
                           gen_circle((800,828),80,75)))

    return points

def gen_dbscan_dataset2():
    """Generate dataset similar to the second dataset in DBSCAN paper

    This function generates a dataset with four non-convex clusters

    Parameters:

    :return: list of points (each point is a tuple)
    """
    shape1 = [ ((692,374),(758,254)),
               ((758,254),(856,352)) ]
    width1 = 40
    shape2 = [ ((476,708),(674,564)),
               ((674,564),(908,642)),
               ((908,642),(930,472)) ]
    width2 = 40
    shape3 = [ ((350,440),(274,466)),
               ((274,466),(244,358)),
               ((244,358),(306,330)) ]
    width3 = 30

    shape4 = [ ((90,384),(114,280)),
               ((114,280),(306,148)),
               ((306,148),(446,242)),
               ((446,242),(522,384)),
               ((522,384),(522,472)),
               ((522,472),(316,702)),
               ((316,702),(300,768)),
               ((300,768),(380,834)),
               ((380,834),(564,806)),
               ((564,806),(660,706)) ]
    width4 = 40

    points=np.concatenate((gen_shape(shape1,width1,100),
                           gen_shape(shape2,width2,200),
                           gen_shape(shape3,width3,100),
                           gen_shape(shape4,width4,400)))

    return points

def gen_dbscan_dataset3():
    """Generate dataset similar to the third dataset in DBSCAN paper

    This function generates a dataset with two "shape-like" clusters
    and two circular ones, plus 10% of the points are background noise

    Parameters:

    :return: list of points (each point is a tuple)
    """
    shape1 = [ ((340,760),(645,750)),
               ((645,750),(715,660)),
               ((715,660),(810,625)),
               ((645,750),(692,810)),
               ((692,810),(787,845)),
               ((787,845),(848,857)) ]
    width1 = 30
    shape2 = [ ((781,746),(890,742)) ]
    width2 = 30

    points=np.concatenate((gen_shape(shape1,width1,300),
                           gen_shape(shape2,width2,100),
                           gen_circle((190,400),70,100),
                           gen_circle((680,300),125,200),
                           gen_square((500,500),500,70)))

    return points

class TestGenDbscanDataset(unittest.TestCase):
    def test_gen_shape(self):
        print('\nRunning test_gen_shape:')

        segmentlist=[((25,50), (65,100)),((65,100),(30,85))]
        shape=gen_shape(segmentlist,1,1000)
        plt.figure(1)
        plt.scatter(shape[:,0], shape[:,1])
        plt.show()
        self.assertEqual(1,1)

    def test_gen_circle(self):
        print('\nRunning test_gen_circle:')

        circle=gen_circle((500,500),315,1000)
        plt.figure(1)
        plt.scatter(circle[:,0], circle[:,1])
        plt.show()
        self.assertEqual(1,1)

    def test_gen_square(self):
        print('\nRunning test_gen_square:')

        square=gen_square((500,500),50,10000)
        plt.figure(1)
        plt.scatter(square[:,0], square[:,1])
        plt.show()
        self.assertEqual(1,1)

    def test_gen_dbscan_dataset1(self):
        print('\nRunning test_gen_dbscan_dataset1:')

        dbscan_ds1=gen_dbscan_dataset1()
        plt.figure(1)
        plt.scatter(dbscan_ds1[:,0], dbscan_ds1[:,1])
        plt.show()

        self.assertEqual(1,1)

    def test_gen_dbscan_dataset2(self):
        print('\nRunning test_gen_dbscan_dataset2:')

        dbscan_ds2=gen_dbscan_dataset2()
        plt.figure(1)
        plt.scatter(dbscan_ds2[:,0], dbscan_ds2[:,1])
        plt.show()

        self.assertEqual(1,1)

    def test_gen_dbscan_dataset3(self):
        print('\nRunning test_gen_dbscan_dataset3:')

        dbscan_ds3=gen_dbscan_dataset3()
        plt.figure(1)
        plt.scatter(dbscan_ds3[:,0], dbscan_ds3[:,1])
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
