#!/usr/bin/env python3
from pytorchYolo.detector import YoloLiveVideoStream
from pytorchYolo.argLoader import ArgLoader
from ampProc.amp_common import ampCommon
import cv2
import glob
import yaml
from time import sleep
from os.path import dirname, abspath
from ampProc.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
import numpy as np


def draw_circles(img, square, radius=10, color=(255, 0, 0)):
    for sq in square:
        if sq is not None:
            lower_left = (sq.lower_x, sq.lower_y)
            upper_left = (sq.lower_x, sq.upper_y)
            upper_right = (sq.upper_x, sq.upper_y)
            lower_right = (sq.upper_x, sq.lower_y)
            circles = [lower_left, upper_left, upper_right, lower_right]

            for circle in circles:
                print(circle)
                cv2.circle(img, circle, radius, color, thickness=-1)


def main(detector):
    with open(args.calibration_yaml, 'r') as stream:
        calibration_loader = yaml.safe_load(stream)
        
    loader = Loader()
    loader.load_params_from_file(calibration_loader)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
    manta1 = sorted(glob.glob(args.images + "/Manta1/*.jpg"))
    manta2 = sorted(glob.glob(args.images + "/Manta2/*.jpg"))

    images = []
    for fname1, fname2 in zip(manta1, manta2):
        images.append((fname1, fname2))
    ac = ampCommon()
    ac.display = False

    img1 = cv2.imread(images[0][0])
    im_size = img1.shape[:2]
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader, im_size)
    # Get rectification maps
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            EI_loader.paramaters.K1, EI_loader.paramaters.d1,
            EI_loader.paramaters.K2, EI_loader.paramaters.d2,
            EI_loader.paramaters.im_size, EI_loader.paramaters.R,
            EI_loader.paramaters.t)
    new_k2 = np.eye(3)
    EI_loader.paramaters.d2[4] = 0
    new_k2, roi = cv2.getOptimalNewCameraMatrix(
        EI_loader.paramaters.K2, EI_loader.paramaters.d2, im_size, 1, im_size)
    # Compute rectification maps
    map1, map2 = cv2.initUndistortRectifyMap(
            EI_loader.paramaters.K2, EI_loader.paramaters.d2, R2,
            new_k2, im_size, cv2.CV_32FC1)

    for i in range(len(images)):
        frame1, frame2 = ac.find_date(images, i)
        if frame1 is not None and frame2 is not None:
            img1 = cv2.imread(frame1)
            img2 = cv2.imread(frame2)
            # rectify image 2
            img2 = cv2.remap(img2, map1, map2, cv2.INTER_LINEAR)

            if img1 is None or img2 is None:
                continue

            detection1, sq1 = detector.stream_img(
                            img1, frame1.split('/')[-1],
                            display_name_append="1")
            detection2, sq2 = detector.stream_img(
                            img2, frame2.split('/')[-1],
                            display_name_append="2")

            if detection1 and detection2:
                draw_circles(img1, sq1)
                draw_circles(img2, sq2)
                draw_circles(img1, sq2, color=(0, 0, 255))
                cv2.imshow("frame1", img1)
                cv2.imshow("frame2", img2)
                cv2.waitKey(0)

            sleep(pause)


if __name__ == '__main__':
    argLoader = ArgLoader()
    argLoader.parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/cfg/calibration.yaml")
    args = argLoader.args  # parse the command line arguments
    pause = 0.01

    detector = YoloLiveVideoStream(args)

    main(detector)
