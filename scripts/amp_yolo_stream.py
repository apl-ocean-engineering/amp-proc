#!/usr/bin/env python3

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""

from pytorchYolo.detector import YoloLiveVideoStream
from pytorchYolo.argLoader import ArgLoader
from ampProc.amp_common import ampCommon, CvSquare
import cv2

import glob
import yaml
from os.path import dirname, abspath
from ampProc.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
import sys
from shapely.geometry import Polygon
import copy
import time
import signal

x_motion = 1

name1 = "img1"
name2 = "img2"

# cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
# cv2.namedWindow(name2, cv2.WINDOW_NORMAL)

overlap_threshold = 0.4

min_det_objs = 6
min_shared_objs = 6

sq_x_extend = 0
sq_y_extend = 50


def sigint_handler(signum, frame):
    """
    Exit system if SIGINT
    """
    sys.exit()


def parse_waitKey(k):
    if k == 113:  # q
        sys.exit()


def draw_images(img1, img2, name1=name1, name2=name2, wait=0):
    cv2.imshow(name1, img1)
    cv2.imshow(name2, img2)

    k = cv2.waitKey(wait)
    parse_waitKey(k)


def test_square_overlap(sq2, square1_list, img1=None, img2=None):
    # draw_circles(img1, sq2, color=(0,0,255))
    for sq1 in square1_list:
        # draw_circles(img1, sq1)
        # draw_images(img1, img2, wait=1)
        p1 = construct_polygon(sq1)
        p2 = construct_polygon(sq2)

        if p1.intersects(p2):

            max_area = max(p1.area, p2.area)
            overlap_over_area = p1.intersection(p2).area/max_area
            # print(overlap_over_area)
            if overlap_over_area > overlap_threshold:
                return True, sq1

    return False, None


def deconstruct_square(sq):
    lower_left = (sq.lower_x, sq.lower_y)
    upper_left = (sq.lower_x, sq.upper_y)
    upper_right = (sq.upper_x, sq.upper_y)
    lower_right = (sq.upper_x, sq.lower_y)
    corners = [lower_left, upper_left, upper_right, lower_right]

    return corners


def construct_polygon(square):
    corners = deconstruct_square(square)

    return Polygon([corners[0], corners[1], corners[2], corners[3]])


def draw_circles(img, sq, radius=10, color=(255, 0, 0)):
    circles = deconstruct_square(sq)
    for circle in circles:
        cv2.circle(img, circle, radius, color, thickness=-1)


def build_squares(im_size, yolo_squares):
    squares = []
    for ysq in yolo_squares:
        if ysq is not None:
            sq = CvSquare(im_size)
            sq.set_from_YOLO_square(ysq)
            sq.extend_square(sq_x_extend, sq_y_extend)
            squares.append(sq)

    return squares


def get_detection_squares(detector, img, frame, display_name_append="1"):
    detection, yolo_squares = detector.stream_img(
                    img, frame.split('/')[-1],
                    display_name_append=display_name_append)

    squares = build_squares(img.shape[0:2], yolo_squares)

    return detection, squares


def check_stereo_detection(squares1, squares2, img1=None, img2=None,
                                        show_YOLO=False, args=None):
    for sq2 in squares2:
        time_init = time.time()
        sq2_init = copy.deepcopy(sq2)
        # draw_circles(img2, sq2)
        # Search forward
        valid = sq2.move_square(motion=1)
        while valid:
            overlap, sq1 = test_square_overlap(
                    sq2, squares1,
                    img1=img1, img2=img2)
            if overlap:
                if show_YOLO:
                    if args is not None:
                        draw_circles(img1, sq1)
                        draw_circles(img2, sq2_init)
                        draw_images(img1, img2, wait=0)
                    else:
                        print("NO ARGS PASSED")
                return True
            valid = sq2.move_square()

        # Search forward
        valid = sq2.move_square(motion=-1)
        while valid:
            overlap, sq1 = test_square_overlap(
                    sq2, squares1,
                    img1=img1, img2=img2)
            if overlap:
                if show_YOLO:
                    if args is not None:
                        draw_circles(img1, sq1)
                        draw_circles(img2, sq2_init)
                        draw_images(img1, img2, wait=0)
                    else:
                        print("NO ARGS PASSED")
                return True
            valid = sq2.move_square()

        print("time elapsed", time.time() - time_init)

    return False


def load_undistort_rectify_image(fname, K1, d1, map1, map2):
    img = cv2.imread(fname)
    raw_img = copy.deepcopy(img)

    # undistort images
    img = cv2.undistort(img, K1, d1)

    # rectify images
    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    return img, raw_img


def detection(detection1, detection2, squares1, squares2, img1, img2):

    if len(squares1) > min_det_objs or len(squares2) > min_det_objs:
        return True

    if detection1 and detection2:
        if len(squares1) > min_shared_objs and len(squares2) > min_shared_objs:
            return True

        stereo_detection = check_stereo_detection(
            squares1, squares2, img1=img1, img2=img2,
            show_YOLO=False, args=args)

        if stereo_detection:
            return True


def main(args, detector):
    with open(args.calibration_yaml, 'r') as stream:
        calibration_loader = yaml.safe_load(stream)

    loader = Loader()
    loader.load_params_from_file(calibration_loader)

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
    if EI_loader.paramaters.R1 is None or EI_loader.paramaters.R1 is None:
        EI_loader.calculate_rectification_matracies()

    # Load rectification maps
    map1_1, map1_2 = cv2.initUndistortRectifyMap(
            EI_loader.paramaters.K1, EI_loader.paramaters.d1,
            EI_loader.paramaters.R1, EI_loader.paramaters.P1[0:3, 0:3],
            im_size, cv2.CV_32FC1)

    map2_1, map2_2 = cv2.initUndistortRectifyMap(
            EI_loader.paramaters.K2, EI_loader.paramaters.d2,
            EI_loader.paramaters.R2, EI_loader.paramaters.P2[0:3, 0:3],
            im_size, cv2.CV_32FC1)

    for i in range(len(images)):
        signal.signal(signal.SIGINT, sigint_handler)
        try:
            frame1, frame2 = ac.find_date(images, i)
            if frame1 is not None and frame2 is not None:
                img1, raw_img1 = load_undistort_rectify_image(
                    frame1, EI_loader.paramaters.K1, EI_loader.paramaters.d1,
                    map1_1, map1_2)

                img2, raw_img2 = load_undistort_rectify_image(
                    frame2, EI_loader.paramaters.K2, EI_loader.paramaters.d2,
                    map2_1, map2_2)

                if img1 is None or img2 is None:
                    continue

                # draw_images(raw_img1, raw_img2, wait=1)

                detection1, squares1 = get_detection_squares(
                        detector, img1, frame1.split('/')[-1],
                        display_name_append="1")

                detection2, squares2 = get_detection_squares(
                        detector, img2, frame2.split('/')[-1],
                        display_name_append="2")

                obj_found = detection(
                    detection1, detection2, squares1, squares2, img1, img2)

                if obj_found:
                    print("DETECTED")
                    print(frame1)
                    # pass
                    # draw_images(raw_img1, raw_img2, wait=0)
                else:
                    pass
                    # draw_images(raw_img1, raw_img2, wait=1)
        except:
            pass


if __name__ == '__main__':
    argLoader = ArgLoader()
    argLoader.parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(
            __file__))) + "/cfg/calibrationConfig.yaml")
    argLoader.parser.add_argument(
        "--save_path",
        help="Path to save positive detections",
        default=dirname(dirname(abspath(
            __file__))) + "/data/detections.txt")
    args = argLoader.args  # parse the command line arguments
    pause = 0.01

    detector = YoloLiveVideoStream(args)
    detector.display = False

    main(args, detector)
