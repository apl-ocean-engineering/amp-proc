#!/usr/bin/env python3

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from pytorchYolo.detector import YoloLiveVideoStream
from pytorchYolo.argLoader import ArgLoader
from ampProc.amp_common import ampCommon, CvSquare
from ampProc.amp_img_proc import BasePath
import cv2
import datetime
import glob

from os.path import dirname, abspath
from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
import sys
from shapely.geometry import Polygon
import copy
import time
import signal

start_date = ' '

def sigint_handler(signum, frame):
    """
    Exit system if SIGINT
    """
    sys.exit()

def main(args, detector):
    ac = ampCommon()

    save_path = args.save_path
    if save_path[-1] != "/":
        save_path += "/"

    base_directory = args.images
    sub_directories = sorted(glob.glob(base_directory + '*/'))

    count = 0
    # try:
    for _dir in sub_directories:
        date = _dir.split("/")[-2]
        print('date', date)
        print('dir', _dir)
        # Ignore folders that aren't of specific dates
        if date[0:2] == '20':
            beyond = ac.beyond_date(date, start_date)
            if beyond:
                bp = BasePath(_dir)
                for folder in bp.sub_directories:
                    blueview = sorted(glob.glob(folder + "BlueView/*.png"))

                for fname in blueview:
                    last_visted = open(
                         save_path + "last_visted.txt", "a+")
                    last_visted.write(
                       str(count) + "," + folder + "," + fname + '\n')
                    last_visted.close()
                    count += 1
                    signal.signal(signal.SIGINT, sigint_handler)

                    img = cv2.imread(fname)
                    #print(type(img))
                    if img is None:
                        continue

                    detection, yolo_squares = detector.stream_img(
                                    img, fname.split('/')[-1])

                    if detection == True:
                            detection_file = open(
                                save_path + "detection.txt", "a+")
                            write_line = str(count) + "," + fname + '\n'
                            detection_file.write(write_line)
                            detection_file.close()




if __name__ == '__main__':
    argLoader = ArgLoader()

    argLoader.parser.add_argument(
        "--save_path",
        help="Path to save positive detections",
        default=dirname(dirname(abspath(
            __file__))) + "/WAMP_acoustic_review")
            

    args = argLoader.args  # parse the command line arguments
    pause = 0.01

    detector = YoloLiveVideoStream(args)
    detector.display = False

    main(args, detector)
