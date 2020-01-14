#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import argparse
import cv2
import numpy as np
import copy

import sys
from os.path import dirname, abspath
from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
from stereoProcessing.point_identification3 import PointIdentification3D


def calculate_norm_distance(distance):
    """
    Returns array of norm distances between points selected
    """
    norm_distances = []
    for i in range(distance.shape[1]):
        norm_distances.append(np.linalg.norm(distance[0:2, i]))

    return norm_distances


def calculate_distances(pnt4):
    """
    Returns array of distances between corresponding points in order of Clicked
    """
    # Convert to homgenous
    pnt4 /= pnt4[3]
    distance = np.zeros((3, pnt4.shape[1]/2))
    for i in range(0, pnt4.shape[1] - 1, 2):
        distance[0:3, i/2] = np.subtract(pnt4[0:3, i], pnt4[0:3, i+1])

    return distance


def main():
    parser = argparse.ArgumentParser(description="Triangulation of AMP image \
                                     points")
    parser.add_argument("--image_path", help="Path to calibration images",
                    default=dirname(dirname(abspath(__file__))) + "/images")
    parser.add_argument("--img1", help="Path to img1", default="/img1.png")
    parser.add_argument("--img2", help="Path to img2", default="/img2.png")
    parser.add_argument("--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/cfg/calibrationConfig.yaml")
    parser.add_argument("--base_path",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/calibration/")

    args = parser.parse_args()

    img_path = args.image_path
    fname1 = img_path + args.img1
    fname2 = img_path + args.img2
    print(fname1)

    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)

    loader = Loader(base_path=args.base_path)
    loader.load_params_from_file(args.calibration_yaml)

    print("Click on corresponding points in both images to estimate length")
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)
    PI = PointIdentification3D(EI_loader)
    points4D = PI.get_points(copy.copy(img1), copy.copy(img2))
    print("4D points: ", points4D)

    distance = calculate_distances(points4D)
    print("Distance between points", distance)
    print("norm distnaces: ", calculate_norm_distance(distance))

    cv2.destroyAllWindows()
    sys.exit()


if __name__ == '__main__':
    """
    Click on points in 2 images IN ORDER. Click on same number of points, and
    output will display 3D distances and norm distance
    """
    main()
