#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import argparse
import yaml
import cv2
import numpy as np


import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))) + '/lib')

from intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
from point_identification3 import PointIdentification3D
from constans import Constants

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triangulation of AMP image \
                                     points")
    parser.add_argument("--image_path", help="Path to calibration images",
                    default=dirname(dirname(abspath(__file__))) + "/images")
    parser.add_argument("--img1", help="Path to img1", default="/img1.png")
    parser.add_argument("--img2", help="Path to img2", default="/img2.png")
    parser.add_argument("--calibration_yaml", help="Path to calibration yaml \
                            specify path of calibration files",
                            default=dirname(dirname(abspath(__file__))) + "/cfg/calibration.yaml")

    args = parser.parse_args()

    img_path = args.image_path
    fname1 = img_path + args.img1
    fname2 = img_path + args.img2

    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)


    with open(args.calibration_yaml, 'r') as stream:
        calibration_loader = yaml.safe_load(stream)

    loader = Loader()
    loader.load_params_from_file(calibration_loader)

    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader, img1.shape[:2])

    PI = PointIdentification3D(EI_loader)
    # k = 0
    # while k != 113: #q
    #     cv2.imshow(Constants.img1_name, img1)
    #     cv2.imshow(Constants.img2_name, img2)
        # k = cv2.waitKey(0)
    points4D = PI.get_points(img1, img2)
    points4D /= points4D[3]
    print(np.linalg.norm(points4D))

    cv2.destroyAllWindows()
    sys.exit()
