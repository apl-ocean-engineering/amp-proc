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


#start_date = '/media/WAMP/2019_01_08'  # Set null string (' ') to search over full space
# example: start_date = '/media/WAMP/2018_11_17'
start_date = ' '

x_motion = 1

name1 = "frame1"
name2 = "frame2"

cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
cv2.namedWindow(name2, cv2.WINDOW_NORMAL)

overlap_threshold = 0.2

min_det_objs = 6
min_shared_objs = 6

sq_x_extend = 0
sq_y_extend = 50

img_count = 0

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
    # if k == 13:
    #     global img_count
    #     img_count += 1
    #     cv2.imwrite("img1" + str(img_count) +".png", img1)
    #     cv2.imwrite("img2" + str(img_count) +".png", img2)
    parse_waitKey(k)


def test_square_overlap(sq2, square1_list, img1=None, img2=None):
    #draw_circles(img1, sq2, color=(255,0,255))
    #draw_images(img1, img2, wait=1)
    for sq1 in square1_list:
        #draw_circles(img1, sq1)
        #draw_images(img1, img2, wait=1)
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
        draw_circles(img2, sq2)

        #draw_images(img1, img2)
        # Search forward
        valid = sq2.move_square(motion=1)
        while valid:
            #draw_circles(img1, sq2)
            #draw_images(img1, img2, wait=1)

            draw_circles(img1, sq2_init)
            overlap, sq1 = test_square_overlap(
                    sq2, squares1,
                    img1=img1, img2=img2)
            if overlap:
                if show_YOLO:
                    if args is not None:
                        draw_circles(img1, sq1)
                        draw_circles(img2, sq2_init)
                        #draw_images(img1, img2, wait=0)
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
    if img is None:
        return None, None
    if img.shape[0] < 60:
        return None, None
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
        #cv2.imshow("img1", img1)
        #k = cv2.waitKey(0)
        k = 13
        if k == 13:
            stereo_detection = check_stereo_detection(
                squares1, squares2, img1=img1, img2=img2,
                show_YOLO=True, args=args)


            if stereo_detection:
                return True
        else:
            False


def beyond_date(date, start_date):
    if start_date == ' ':
        return True
    else:
        year = int(date.split('_')[0])
        month = int(date.split('_')[1])
        day = int(date.split('_')[2])
        date1 = datetime.date(year=year, month=month, day=day)
        start_date = start_date.split("/")[3]
        start_yr = int(start_date.split('_')[0])
        start_month = int(start_date.split('_')[1])
        start_day = int(start_date.split('_')[2])
        date2 = datetime.date(year=start_yr, month=start_month, day=start_day)
        if time.mktime(date1.timetuple()) > time.mktime(date2.timetuple()):
            return True
        return False


def main(args, detector):
    ac = ampCommon()

    save_path = args.save_path
    if save_path[-1] != "/":
        save_path += "/"

    loader = Loader(args.base_path)
    loader.load_params_from_file(args.calibration_yaml)

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
                    global img_count
                    print(count, folder)
                    manta1 = sorted(glob.glob(folder + "Manta 1/*.jpg"))
                    manta2 = sorted(glob.glob(folder + "Manta 2/*.jpg"))

                    images = []
                    for fname1, fname2 in zip(manta1, manta2):
                        images.append((fname1, fname2))

                    ac.display = False

                    im_size = (2464, 2056)
                    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

                    # Get rectification maps
                    if EI_loader.paramaters.R1 is None or EI_loader.paramaters.R1 is None:

                        EI_loader.calculate_rectification_matracies()

                    # Load rectification maps
                    map1_1, map1_2 = cv2.initUndistortRectifyMap(
                            EI_loader.paramaters.K1, EI_loader.paramaters.d1,
                            EI_loader.paramaters.R1,
                            EI_loader.paramaters.P1[0:3, 0:3], im_size,
                            cv2.CV_32FC1)

                    map2_1, map2_2 = cv2.initUndistortRectifyMap(
                            EI_loader.paramaters.K2, EI_loader.paramaters.d2,
                            EI_loader.paramaters.R2,
                            EI_loader.paramaters.P2[0:3, 0:3], im_size,
                            cv2.CV_32FC1)
                    for i in range(len(images)):
                        time_init = time.time()

                        count += 1

                        signal.signal(signal.SIGINT, sigint_handler)

                        frame1, frame2 = ac.find_date(images, i)
                        if frame1 is not None and frame2 is not None:
                            print(frame1.split('/')[-1], frame2.split('/')[-1])
                            last_visted = open(
                                 save_path + "last_visted.txt", "a+")
                            last_visted.write(
                               str(count) + "," + folder + "," + frame1 + '\n')
                            last_visted.close()
                            img1, raw_img1 = load_undistort_rectify_image(
                                frame1, EI_loader.paramaters.K1,
                                EI_loader.paramaters.d1,
                                map1_1, map1_2)

                            img2, raw_img2 = load_undistort_rectify_image(
                                frame2, EI_loader.paramaters.K2,
                                EI_loader.paramaters.d2,
                                map2_1, map2_2)

                            if img1 is None or img2 is None:
                                continue

                            #draw_images(raw_img1, raw_img2, wait=1)

                            detection1, squares1 = get_detection_squares(
                                    detector, img1, frame1.split('/')[-1],
                                    display_name_append="1")

                            detection2, squares2 = get_detection_squares(
                                    detector, img2, frame2.split('/')[-1],
                                    display_name_append="2")

                            obj_found = detection(
                                detection1, detection2, squares1, squares2,
                                img1, img2)

                            if obj_found:
                                detection_file = open(
                                    save_path + "detection.txt", "a+")
                                write_line = str(count) + "," + frame1 + \
                                      "," + frame2 + '\n'
                                detection_file.write(write_line)
                                detection_file.close()
                            else:
                                pass
                                # draw_images(raw_img1, raw_img2, wait=1)
                            print("total time elapsed", time.time() - time_init)
    # except:
        # pass


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
            __file__))) + "/WETS_stereo_data_review")
    argLoader.parser.add_argument(
        "--base_path", help="Base folder to calibration values",
        default=dirname(dirname(abspath(__file__))) + "/calibration/")

    args = argLoader.args  # parse the command line arguments
    pause = 0.01

    detector = YoloLiveVideoStream(args)
    detector.display = False

    main(args, detector)
