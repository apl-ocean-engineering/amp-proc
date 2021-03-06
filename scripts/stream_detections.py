#!/usr/bin/env python3

import cv2
import os
import time
import argparse
from os.path import dirname, abspath
from ampProc.image_stream import ImageStream


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
    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)


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
    while img_num < len(lines):
        img_file = lines[img_num]
        img_name1 = img_file.split(',')[-2]
        img_name2 = img_file.split(',')[-1].rstrip()
        img1 = cv2.imread(img_name1)
        img2 = cv2.imread(img_name2)
        img_num = IS.stream_images(img1, img2, img_num)

        # if k == 112:  # p
        #     print("PAUSING")
        #     self.wait_time = 0
        #     # return img_num
        #
        # if k == 99:  # c
        #     print("CONTINUING")
        #     self.wait_time = 50
        #     # return img_num
        #
        # if k == 110:  # n
        #     print("NEXT IMAGE")
        #     img_num += 1
        #     # return img_num
        #
        # if k == 117:  # u
        #     print("UNUSRE")
        #     unsure_write_file = open(
        #         interesting_event_name.replace('true', 'unsure'), 'a+')
        #     unsure_write_file.write(img_file.split(',')[-1])
        #     unsure_write_file.close()
        #
        # elif k == 102:  # f
        #     print('FALSE CLASSIFICATION')
        #     ''' False '''
        #     if os.path.exists(write_file_name):
        #         remove_line(write_file_name, img_file.split(',')[-1])
        #
        # elif k == 98:  # b
        #     print('GOING BACKWARDS')
        #     '''
        #     GO BACK
        #     '''
        #     img_num -= 1
        #     # return img_num
        #
        # elif k == 113:  # q
        #     print('EXITING')
        #     ''' Exit '''
        #     exit()
        #
        # elif k == 13:  # Enter
        #     print('INTERESTING EVENT')
        #     ''' TRUE '''
        #     interesting_even_file = open(write_file_name, 'a+')
        #     interesting_even_file.write(
        #     img_file.split(',')[-2] + "," + img_file.split(',')[-1])
        #     interesting_even_file.close()

        save_img_file = open(args.last_image_save_file, 'a+')
        save_img_file.write(str(img_num) + '\n')
        save_img_file.close()




if __name__ == '__main__':
    main()
