#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import cv2
import argparse
from os.path import dirname, abspath

frame_name1 = "frame1"
frame_name2 = "frame2"

cv2.namedWindow(frame_name1, cv2.WINDOW_NORMAL)
cv2.namedWindow(frame_name2, cv2.WINDOW_NORMAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("View detected events")
    parser.add_argument(
        "--detection_path",
        help="Path to save positive detections",
        default=dirname(dirname(abspath(
            __file__))) + "/data/detection.txt")

    args = parser.parse_args()  # parse the command line arguments

    f = open(args.detection_path, 'r')
    for line in f:
        fname1 = line.split(',')[1]
        fname2 = line.split(',')[2].rstrip()
        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        cv2.imshow(frame_name1, img1)
        cv2.imshow(frame_name2, img2)

        cv2.waitKey(1)
