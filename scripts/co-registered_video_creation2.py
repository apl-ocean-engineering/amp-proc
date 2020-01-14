import cv2
import argparse
import glob
from ampProc.amp_common import ampCommon
import numpy as np

def main(images):
    file = open(images, 'r')
    for fnames in file:
        optical = cv2.imread(fnames.split(',')[0])
        sonar = cv2.imread(fnames.split(',')[1].rstrip())
        cv2.imshow('optical', optical)
        cv2.imshow('sonar', sonar)
        cv2.waitKey(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creation of co-registered \
        videos from AMP optical/acoustic data")
    parser.add_argument("--images_file", help="Path to images text file",
                        default="data/co-registered_detections.txt")

    args = parser.parse_args()


    main(args.images_file)
