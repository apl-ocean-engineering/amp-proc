#!/usr/bin/env python3

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from pytorchYolo.detector import YoloLiveVideoStream
from pytorchYolo.argLoader import ArgLoader
import cv2
from os.path import dirname, abspath
from ampProc.stereo_processing import StereoProcessing
from ampProc.amp_img_proc import BasePath
from stereoProcessing.intrinsic_extrinsic import Loader
from stereoProcessing.intrinsic_extrinsic import ExtrinsicIntrnsicLoaderSaver
from ampProc.amp_common import ampCommon
import glob
import time
import numpy as np
import signal
import sys
import copy

import json


class TemporalDetectionSquaresCorrespondance:
    X_EXTEND = 30
    Y_EXTEND = 30

    OVERLAP_WEIGHT = 1.0
    AREA_WEIGHT = 1.0
    TRAVEL_WEIGHT = 1.0/100

    MAX_TRAVEL_DIST = 800

    def __init__(self, fname, SP):
        self.fname = fname
        self.SP = SP
        self.squares_list = []
        self.prev_avg_lst = []

    def append_squares(self, sq):
        self.squares_list.append(sq)

    def append_squares_lst(self, sq_lst):
        for sq in sq_lst:
            self.squares_list.append(sq)

    def find_corresponding_points(self, sq):
        sq = copy.deepcopy(sq)
        overlap_scores = self._overlap(sq)
        area_scores = self._area(sq)
        travel_scores = self._travel_scores(sq)

        min_score = 1000
        return_sq = None
        for i in range(len(self.squares_list)):
            if overlap_scores[i] == -1:
                overlap_scores[i] = 0
                #continue
            if travel_scores[i] == -1:
                print("Traveled too far for overlap")
                continue
            total_score = self.OVERLAP_WEIGHT*overlap_scores[i] + \
                self.AREA_WEIGHT*area_scores[i] + \
                self.TRAVEL_WEIGHT*travel_scores[i]
            # print(self.OVERLAP_WEIGHT*overlap_scores[i],
            #       self.AREA_WEIGHT*area_scores[i],
            #       self.TRAVEL_WEIGHT*travel_scores[i])

            if total_score < min_score:
                min_score = total_score
                return_sq = self.squares_list[i]

            #print("total_score", total_score)
        return return_sq

    def _travel_scores(self, sq):
        scores = []
        for i, sq_t1 in enumerate(self.squares_list):
            distance_travled = sq_t1.distance_compairson(sq)
            if distance_travled > self.MAX_TRAVEL_DIST:
                scores.append(-1)
            else:
                max_dist = max(sq_t1.img_size[0], sq_t1.img_size[1])
                scores.append(distance_travled/max_dist)

        return scores

    def _area(self, sq):
        scores = []
        for i, sq_t1 in enumerate(self.squares_list):
            sq_t1 = copy.deepcopy(sq_t1)
            sq_t1.extend_square(self.X_EXTEND, self.Y_EXTEND)
            sq.extend_square(self.X_EXTEND, self.Y_EXTEND)
            poly_t1 = self.SP.construct_polygon(sq_t1)
            poly_t = self.SP.construct_polygon(sq)
            area_t1 = poly_t1.area
            area_t = poly_t.area
            if area_t1 > area_t:
                area_ratio = 1 - area_t/area_t1
            else:
                area_ratio = 1 - area_t1/area_t
            scores.append(area_ratio)

        return scores

    def _overlap(self, sq):
        scores = []
        for i, sq_t1 in enumerate(self.squares_list):
            sq_t1 = copy.deepcopy(sq_t1)
            sq_t1.extend_square(self.X_EXTEND, self.Y_EXTEND)
            sq.extend_square(self.X_EXTEND, self.Y_EXTEND)
            poly_t1 = self.SP.construct_polygon(sq_t1)
            poly_t = self.SP.construct_polygon(sq)
            if not poly_t1.intersects(poly_t):
                scores.append(-1)
                continue
            max_area = max(poly_t.area, poly_t1.area)
            overlap_over_area = 1 - max_area/min(poly_t.area, poly_t1.area)#1 - poly_t.intersection(poly_t1).area / max_area

            scores.append(overlap_over_area)

        return scores
        # ind = -1
        # max_overlap_over_area = -1
        # for i, sq_t1 in enumerate(self.squares_list):
        #     sq_t1.extend_square(30, 30)
        #     sq.extend_square(30, 30)
        #     poly_t1 = self.SP.construct_polygon(sq_t1)
        #     poly_t = self.SP.construct_polygon(sq)
        #     if not poly_t1.intersects(poly_t):
        #         continue
        #     max_area = max(poly_t.area, poly_t1.area)
        #     overlap_over_area = poly_t.intersection(poly_t1).area / max_area
        #     if overlap_over_area > max_overlap_over_area:
        #         max_overlap_over_area = overlap_over_area
        #         ind = i
        # if ind == -1:
        #     return None
        # return self.squares_list[ind]


def sigint_handler(signum, frame):
    """
    Exit system if SIGINT
    """
    sys.exit()


def order_points(sq1, sq2):
    points1 = [(sq1.lower_x, sq1.lower_y), (sq1.lower_x, sq1.upper_y),
               (sq1.upper_x, sq1.upper_y), (sq1.upper_x, sq1.lower_y)]
    points2 = [(sq2.lower_x, sq2.lower_y), (sq2.lower_x, sq2.upper_y),
               (sq2.upper_x, sq2.upper_y), (sq2.upper_x, sq2.lower_y)]

    points1 = np.asarray(
        [(np.float64(val[0]),
            np.float64(val[1])) for val in points1], dtype=np.float64)
    points2 = np.asarray(
        [(np.float64(val[0]),
            np.float64(val[1])) for val in points2], dtype=np.float64)

    return np.transpose(points1), np.transpose(points2)


def find_points(EI_loader, sq1, sq2):
    points1, points2 = order_points(sq1, sq2)

    points4d = cv2.triangulatePoints(
        EI_loader.paramaters.P1,
        EI_loader.paramaters.P2, points1, points2)

    points4d /= points4d[3]

    return points4d


def run_folders(args, detector):
    save = True

    start_date = ' '
    ac = ampCommon(time_delay_allowed=0.05)

    SP = StereoProcessing(args, detector)

    save_path = args.save_path
    if save_path[-1] != "/":
        save_path += "/"

    loader = Loader(args.base_path)
    loader.load_params_from_file(args.calibration_yaml)

    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

    base_directory = args.images
    sub_directories = sorted(glob.glob(base_directory + '*/'))

    prev_temporal_sq1 = None
    prev_temporal_sq2 = None

    im_size = (int(EI_loader.paramaters.im_size[0]), int(EI_loader.paramaters.im_size[1]))

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

    count = 0

    # try:

    for _dir in sub_directories:
        date = _dir.split("/")[-2]
        print('date', date)
        print('dir', _dir)

        data = {}
        if date[0:2] == '20':

            beyond = ac.beyond_date(date, start_date)
            if beyond:

                bp = BasePath(_dir)
                for folder in bp.sub_directories:
                    global img_count

                    manta1 = sorted(glob.glob(folder + "Manta 1/*.jpg"))
                    manta2 = sorted(glob.glob(folder + "Manta 2/*.jpg"))

                    images = []
                    prev_avg = np.asarray([0, 0, 0])
                    for fname1, fname2 in zip(manta1, manta2):
                        images.append((fname1, fname2))

                    for i in range(len(images)):
                        count += 1

                        signal.signal(signal.SIGINT, sigint_handler)
                        fname1, fname2 = ac.find_date(images, i)
                        if fname1 is not None and fname2 is not None:
                            #data['date'] = {'date' : fname1.split("/")[-1]}
                            # time_diff = ac.relative_time_diff(fname1, fname2)
                            time_init = time.time()
                            img1 = cv2.imread(fname1)
                            img2 = cv2.imread(fname2.rstrip())

                            #
                            img1, raw_img1 = SP.load_undistort_rectify_image(
                                img1, EI_loader.paramaters.K1,
                                EI_loader.paramaters.d1,
                                map1_1, map1_2)

                            img2, raw_img2 = SP.load_undistort_rectify_image(
                                img2, EI_loader.paramaters.K2,
                                EI_loader.paramaters.d2,
                                map2_1, map2_2)
                            #
                            detections, img1, img2 = SP.find_correspondance(
                                img1, img2=img2)
                            _img1 = copy.deepcopy(img1)
                            # print("detection time", time.time() - time_init)

                            if detections != []:
                                temporal_square1 = \
                                    TemporalDetectionSquaresCorrespondance(
                                        fname1, SP)
                                temporal_square2 = \
                                    TemporalDetectionSquaresCorrespondance(
                                        fname2, SP)

                                # print(len(detections))
                                square_count = 0
                                for moved_sq1, moved_sq2, sq1, sq2 in detections:
                                    if sq1 is not None and sq2 is not None:
                                        square_count +=1
                                        SP.draw_circles(img1, sq1)
                                        SP.draw_circles(img2, sq2)
                                        #SP.draw_images(img1, img2, wait=0)



                                        points4d = find_points(EI_loader,
                                            sq1, sq2)
                                        points4dT = np.transpose(points4d)
                                        avg = np.mean(points4dT, axis=0)[:3]
                                        detection_name = 'detection%s_%s' % (count, square_count)
                                        data[detection_name] = {
                                            'fname1': str(fname1),
                                            'fname2': str(fname2),
                                            'sq1_pose': sq1.get_square(),
                                            'sq2_pose': sq2.get_square(),
                                            'detection_size': avg.tolist(),
                                            'detection_loc': points4dT[:3].tolist()}
                                        #data = str(date) + ',' + fname1 + ',' \
                                        #    + ',' + fname2 + str(avg)

                                        avg_size = round(np.mean(avg), 2)
                                        motion_text = "Size is: %s m" \
                                                % (avg_size)
                                        cv2.putText(img1,
                                            motion_text,
                                            (sq1.lower_x,
                                             sq1.lower_y - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                                            (0, 0, 255))

                                        #SP.draw_images(img1, img2, wait=0)

                                        size_p1 = points4d[:, 0]
                                        size_p2 = points4d[:, 2]

                                        #print("size", np.linalg.norm(
                                        #   np.subtract(size_p1, size_p2)))

                                        temporal_square1.append_squares(sq1)
                                        temporal_square2.append_squares(sq2)

                                        #print(square_count, len(temporal_square1.squares_list))

                                        if prev_temporal_sq1 is not None:

                                            #print(square_count, len(prev_temporal_sq1.squares_list))
                                            #print("prev_score1")
                                            prev_sq1 = prev_temporal_sq1.\
                                                find_corresponding_points(sq1)
                                            #print("prev_score2")
                                            prev_sq2 = prev_temporal_sq2.\
                                                find_corresponding_points(sq2)
                                            time_elap = ac.relative_time_diff(
                                                prev_temporal_sq1.fname, fname1,
                                                )
                                            #print('time_elap', time_elap)
                                            if abs(time_elap) > 2:
                                                prev_temporal_sq1 = None
                                                prev_temporal_sq2 = None
                                                prev_sq1 = None
                                                prev_sq2 = None

                                            if prev_sq1 is not None and prev_sq2 is not None:

                                                SP.draw_circles(img1, prev_sq1,
                                                                color=(0, 255, 0))

                                                SP.draw_circles(img2, prev_sq2,
                                                                color=(0, 255, 0))

                                                prev_points4d = find_points(
                                                    EI_loader, prev_sq1, prev_sq2)
                                                prev_points4dT = np.transpose(
                                                    prev_points4d)
                                                prev_avg = np.mean(prev_points4dT,
                                                    axis=0)[:3]

                                                motion = np.subtract(avg, prev_avg)

                                                data[detection_name]['motion'] = motion.tolist()

                                                speed = np.divide(motion,
                                                    time_elap)
                                                avg_speed = round(
                                                    np.linalg.norm(speed), 2)
                                                #print("speed", speed)
                                                #print("avg speed", avg_speed)

                                                speed_text = "Speed is: %s m/s" \
                                                    % (avg_speed)


                                                cv2.putText(img1,
                                                    speed_text,
                                                    (sq1.lower_x + 30,
                                                     sq1.lower_y + 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                    255)

                                        SP.draw_images(img1, img2, wait=1)



                                prev_temporal_sq1 = temporal_square1
                                prev_temporal_sq2 = temporal_square2
                file = "size_speed_test/%s.json" % (date)
                with open(file, 'w') as outfile:
                    json.dump(data, outfile)


def run_txt_file(args, detector):
    start_date = ' '
    ac = ampCommon(time_delay_allowed=0.05)

    SP = StereoProcessing(args, detector)

    save_path = args.save_path
    if save_path[-1] != "/":
        save_path += "/"

    loader = Loader(args.base_path)
    loader.load_params_from_file(args.calibration_yaml)

    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

    prev_temporal_sq1 = None
    prev_temporal_sq2 = None

    count = 0

    txt_file = args.txt_file.rstrip()
    f = open(txt_file, 'r')
    for line in f:
        line = line.split(',')
        fname1 = line[0]
        fname2 = line[1].rstrip()
        print(fname1, fname2)

        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)

        detections, img1, img2 = SP.find_correspondance(
            img1, img2=img2)
        # _img1 = copy.deepcopy(img1)

        if detections != []:
            temporal_square1 = \
                TemporalDetectionSquaresCorrespondance(
                    fname1, SP)
            temporal_square2 = \
                TemporalDetectionSquaresCorrespondance(
                    fname2, SP)

            for moved_sq1, moved_sq2, sq1, sq2 in detections:
                if sq1 is not None and sq2 is not None:
                    SP.draw_circles(img1, sq1)
                    SP.draw_circles(img2, sq2)

                    SP.draw_images(img1, img2, wait=0)
                    #
                    #
                    # points4d = find_points(EI_loader,
                    #                        sq1, sq2)
                    # points4dT = np.transpose(points4d)
                    # avg = np.mean(points4dT, axis=0)[:3]
                    #
                    # size_p1 = points4d[:, 0]
                    # size_p2 = points4d[:, 2]
                    #
                    # print("size", np.linalg.norm(
                    #     np.subtract(size_p1, size_p2)))
                    #
                    # temporal_square1.append_squares(sq1)
                    # temporal_square2.append_squares(sq2)
                    #
                    # if prev_temporal_sq1 is not None:
                    #     prev_sq1 = prev_temporal_sq1.\
                    #         find_corresponding_points(sq1)
                    #     prev_sq2 = prev_temporal_sq2.\
                    #         find_corresponding_points(sq2)
                    #
                    #     if prev_sq1 is None or prev_sq2 is None:
                    #         continue
                    #     # SP.draw_circles(prev_img, prev_sq1)
                    #
                    #     SP.draw_circles(img1, prev_sq1,
                    #                     color=(0, 255, 0))
                    #
                    #
                    #     prev_points4d = find_points(
                    #         EI_loader, prev_sq1, prev_sq2)
                    #     prev_points4dT = np.transpose(
                    #         prev_points4d)
                    #     prev_avg = np.mean(prev_points4dT,
                    #                        axis=0)[:3]
                    #
                    #     motion = np.subtract(avg, prev_avg)
                    #     time_elap = ac.relative_time_diff(
                    #         fname1,
                    #         prev_temporal_sq1.fname)
                    #
                    #     speed = np.divide(motion,
                    #                       time_elap)
                    #     avg_speed = round(
                    #         np.linalg.norm(speed), 2)
                    #
                    #     print("speed", speed)
                    #     print("avg speed", avg_speed)
                    #
                    #     speed_text = "Speed is: %s m/s" % (avg_speed)
                    #
                    #     cv2.putText(img1,
                    #                 speed_text,
                    #                 (sq1.lower_x + 30,
                    #                  sq1.lower_y + 30),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 2,
                    #                 255)
                    #
                    #     SP.draw_images(img1, img2, wait=10)

            prev_temporal_sq1 = temporal_square1
            prev_temporal_sq2 = temporal_square2


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

    argLoader.parser.add_argument(
        "--run_txt_file", help="Base folder to calibration values",
        default=False)

    argLoader.parser.add_argument(
        "--txt_file", help="Base folder to calibration values",
        default=dirname(dirname(abspath(__file__))) + "/data/interesting_events.txt")

    args = argLoader.args  # parse the command line arguments
    pause = 0.01

    detector = YoloLiveVideoStream(args)
    detector.display = False
    detector.write_images = False

    if args.run_txt_file:
        run_txt_file(args, detector)

    else:
        run_folders(args, detector)
