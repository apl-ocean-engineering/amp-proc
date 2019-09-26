#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu

PointClick object to help with point-and-click image point getting for 2 Manta
images. Helper functions will return corresponding image points
(assuming in-order clicking)
"""

import sys

import cv2
import numpy as np

from constans import Constants



class PointClick(object):
    """
    Class to help determine transformation between frames in two 3G-AMP cameras

    Attributes:
        -x#_points, y#_points (list<float>): 4 lists containing corresponding
            points in each camera frame
        -image1, image2 (np.mat<float>): Images
        -m1_subdirectories, m2_subdirectories (list<str>): List
            containing image paths
    Methods:

    """

    def __init__(self):
        """
        Args:
            images_path(str): Path pointing to location of images
        """

        self.x1_points = []
        self.y1_points = []
        self.x2_points = []
        self.y2_points = []
        self.image1 = np.zeros([0, 0])
        self.image2 = np.zeros([0, 0])

        self.img1 = np.zeros((3,3))
        self.img2 = np.zeros((3,3))


        cv2.namedWindow(Constants.img1_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(Constants.img2_name, cv2.WINDOW_NORMAL)
        # Define mouse callback functions
        cv2.setMouseCallback(Constants.img1_name, self._mouse_click1)
        cv2.setMouseCallback(Constants.img2_name, self._mouse_click2)

    def corresponding_image_points(self, img1, img2):
        """
        Determine coressponding image points between the frames

        Will display two WAMP images. User must click on identical point
        in two frames. x#_points, and y#_points will populate as the user
        clicks on points
        """
        self.img1 = img1
        self.img2 = img2
        # Initalzie image windows
        # Loop through all images in subdirectory location
        print("Click on the same point in both images")
        print("Press enter to move finish")
        print("Press q to quit")
        # Show images
        cv2.imshow(Constants.img1_name, self.img1)
        cv2.imshow(Constants.img2_name, self.img2)
        k = 0
        while k != 13: #enter
            k = cv2.waitKey(0)
            if k == 113: #q
                cv2.destroyAllWindows()
                sys.exit()
        cv2.destroyAllWindows()

    def get_points(self):
        """
        Return corresponding image points

        Return:
            points1, points2 (list<tuple<float>>): Corresponding points
        """
        points1, points2 = self._image_points()

        return points1, points2

    def _image_points(self):
        """
        Organize image points into two lists of corresponding tuples

        Return:
            pnts1, pnts2 (list<tuple<float>>): Corresponding points
        """
        # Check that points clicked are equal
        if len(self.x1_points) != len(self.x2_points):
            raise AttributeError("Unequal Points Clicked")
        # Organize points
        pnts1 = np.array([[self.x1_points[0]], [self.y1_points[0]]])
        pnts2 = np.array([[self.x2_points[0]], [self.y2_points[0]]])
        for i in range(1, len(self.x1_points)):
            pnts1 = np.concatenate((pnts1,
                [[self.x1_points[i]], [self.y1_points[i]]]), axis=1)
            pnts2 = np.concatenate((pnts2,
                [[self.x2_points[i]], [self.y2_points[i]]]), axis=1)

        # Must be float 64s to work in OpenCV
        pnts1 = np.float64(pnts1)
        pnts2 = np.float64(pnts2)

        return pnts1, pnts2

    def _mouse_click1(self, event, x, y, flags, param):
        """
        Callback function for mouse click event on image1 frame

        Places clicked points into x1_ and y1_points lists
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1_points.append(x)
            self.y1_points.append(y)
            # Draw circle where clicked
            cv2.circle(self.img1, (x, y), 7, (255, 0, 0), -1)
            cv2.imshow(Constants.img1_name, self.img1)

    def _mouse_click2(self, event, x, y, flags, param):
        """
        Callback function for mouse click event on image2 frame

        Places clicked points into x2_ and y2_points lists
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x2_points.append(x)
            self.y2_points.append(y)
            # Draw circle where clicked
            cv2.circle(self.img2, (x, y), 7, (255, 0, 0), -1)
            cv2.imshow(Constants.img2_name, self.img2)
