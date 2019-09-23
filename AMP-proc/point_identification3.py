#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from img_point_click import PointClick
import cv2


class PointIdentification3D:

    def __init__(self, EI_loader):
        """
        Inputs:
            params: Paramters object
        """
        if EI_loader.paramaters.P1 is None or EI_loader.paramaters.P2 is None:
            EI_loader.calculate_projection_matracies()

        self.EI_loader = EI_loader
        self.params = EI_loader.paramaters

        self.pc = PointClick()

    def get_points(self, img1, img2):
        self.pc.corresponding_image_points(img1, img2)
        points1, points2 = self.pc.get_points()
        points4d = cv2.triangulatePoints(self.params.P1, self.params.P2,
                                                        points1, points2)

        return points4d