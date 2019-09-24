#!/usr/bin/env python2.7

"""
Code to read yaml 1.0 docs and convert them to yaml 1.1 calibration paramters
(R,T, K1, etc.)

@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import cv2
import numpy as np

import argparse
from os.path import dirname, abspath

def get_save_node(f, name, save_name):
    mat = f.getNode(name).mat()
    np.savetxt(save_name, mat,
                           fmt="%1.3f", delimiter=",")

def parse_file_storage(args):
    f = cv2.FileStorage(args.calibration_yaml, cv2.FILE_STORAGE_READ)
    get_save_node(f, "K1", args.save_path + "camera1/intrinsic_matrix.csv")
    get_save_node(f, "K2", args.save_path + "camera2/intrinsic_matrix.csv")
    get_save_node(f, "D1", args.save_path + "camera1/distortion_coeffs.csv")
    get_save_node(f, "D2", args.save_path + "camera2/distortion_coeffs.csv")
    get_save_node(f, "R", args.save_path + "R.csv")
    get_save_node(f, "T", args.save_path + "t.csv")
    get_save_node(f, "E", args.save_path + "E.csv")
    get_save_node(f, "F", args.save_path + "F.csv")
    get_save_node(f, "R1", args.save_path + "R1.csv")
    get_save_node(f, "R2", args.save_path + "R2.csv")
    get_save_node(f, "P1", args.save_path + "P1.csv")
    get_save_node(f, "P2", args.save_path + "P2.csv")
    get_save_node(f, "Q", args.save_path + "Q.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save calibration YAML files")
    parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/cfg/cam_stereo.yml")
    parser.add_argument(
            "--save_path",
            help="Path to save calibration files",
            default=dirname(dirname(abspath(__file__))) + "/calibration/")
    args = parser.parse_args()

    print(args.calibration_yaml)
    parse_file_storage(args)
