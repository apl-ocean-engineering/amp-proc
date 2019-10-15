#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import argparse
import cv2
from os.path import dirname, abspath


def pull_detections(filename, save_path, save_name, test_num = 5):
    """
    Test num: 1/N quantatiy of images to save
    """
    f = open(filename, "r")
    count = 0

    if save_path[-1] != '/':
        save_path + '/'

    for line in f:
        #print(count)
        manta1_img = cv2.imread(line.split(',')[1].rstrip())
        manta2_img = cv2.imread(line.split(',')[2].rstrip())
        #
        manta1_extension = line.split(',')[1].split('/')[-1]
        manta2_extension = line.split(',')[2].split('/')[-1]
        #
        #
        if count % test_num != 0:
             manta1_save_path = save_path + save_name + "Manta1_train"
        else:
             manta1_save_path = save_path + save_name + "Manta1_test"
        #
        manta2_save_path = manta1_save_path.replace('Manta1', 'Manta2')
        #
        print(manta1_save_path, manta2_save_path)
        cv2.imwrite(manta1_save_path + "/" + manta1_extension, manta1_img)
        cv2.imwrite(manta2_save_path + "/" + manta2_extension, manta2_img)

        count+=1





if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pull detection data from WAMP")
    parser.add_argument("base_save_path")
    parser.add_argument(
        "--detections_file_path", help="Path to detections file",
        default=dirname(dirname(abspath(__file__))) + "/data/detection.txt")
    parser.add_argument(
        "--local_save_name", help="Local folder name",
        default="WAMP_detections_images_")

    args = parser.parse_args()

    pull_detections(args.detections_file_path, args.base_save_path, args.local_save_name)
