#!/usr/bin/env python3

import cv2
import os
import time
import argparse
from os.path import dirname, abspath
from ampProc.image_stream import ImageStream
from ampProc.constans import Constants



def remove_line(write_file_name, img_file):
    write_file = open(write_file_name, 'r')
    file_lines = write_file.readlines()
    write_file.close()
    write_file = open(write_file_name, 'w+')
    for line in file_lines:
        if line != img_file:
            write_file.write(line)
    write_file.close()


def main():
    cv2.namedWindow(Constants.img1_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(Constants.img2_name, cv2.WINDOW_NORMAL)


    parser = argparse.ArgumentParser("Stream detected events")

    # parser = argparse.ArgumentParser("Analyze detected 3G-AMP images")
    data_path = dirname(dirname(abspath(__file__))) + "/data"
    parser.add_argument(
        "--optical_analysis_file",
        help="Text file to load optical detections from",
        default=data_path+"/detection.txt")
    parser.add_argument(
        "--start", help="Quantity to start analysis at", default=0, type=int)
    parser.add_argument(
        "--interesting_event_name",
        help="Text file to write true detections to",
        default=data_path+"/interesting_events.txt")
    parser.add_argument(
        "--last_image_save_file",
        help="Text file to write true last_image_visited",
        default=data_path+"/last_visted_img.txt")
    parser.add_argument(
        "--resume",
        help="Resume classification from last known start location",
        default=False, type=bool)

    args = parser.parse_args()

    interesting_event_name = args.interesting_event_name

    f = open(args.optical_analysis_file, 'r')
    lines = f.readlines()
    if args.resume:
        save_img_file = open(args.last_image_save_file, 'r')
        save_lines = save_img_file.readlines()
        img_num = int(save_lines[-1].rstrip())
        save_img_file.close()
    else:
        img_num = args.start

    print("Press 'Enter' to classify event as true. Press 'p' to pause \
        stream. Press 'b' to go back 1 image if paused. press 'n' to go \
        forward 1 image if paused. Press 'c' to continue stream if paused")
    print("If event is incorrectly classified as true, move back to the image \
        and press 'f' to remove it from the false.")
    print("Press 'q' to quit")
    print("To resume from last location, set --resume True on start. \
        Alternativly, set --start <num> to start from a specific location")

    waitTime = 50
    IS = ImageStream(args.interesting_event_name, wait_time=waitTime)
    # BB = BoundingBoxes()
    while img_num < len(lines):
        img_file = lines[img_num]
        img_name = img_file.split(',')[-1].rstrip()
        img = cv2.imread(img_name)
        label = img_name.replace("image", "label").replace(".jpg", ".txt")
        # BB.display_labels(img, label)

        img_num = IS.display_labels(img, label, img_num)

        print(label)

        save_img_file = open(args.last_image_save_file, 'a+')
        save_img_file.write(str(img_num) + '\n')
        save_img_file.close()


if __name__ == '__main__':
    main()
