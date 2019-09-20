#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import glob
import os
import sys
import signal
import logging
import time
import datetime

import cv2
import numpy as np


def get_paths(directory, flag=False):
    """
    Get list of all files and folders under a specific directory satisfying
    the input form

    Input:
        directory (str): Name of directory in which files should be, PLUS
        the file type, if desired. Must have '*' symbol were searchign will
        take place. It will return all files that fit the
        specified format, filling in for the '*'

        Example 1: path + *.jpg will return all .jpg files in the location
        specified by path

        Example 2: path + */ will return all folders in the location
        specified  by oath

        Example 3: path + * will return all folders AND files in the
        location specified by path

    Return:
        paths (list<strs>): List of strings for each file/folder which
        satisfies the in input style.
        Empty list if no such file/folder exists
    """

    paths = sorted(glob.glob(directory))
    return paths


def sigint_handler(signum, frame):
    """
    Exit system if SIGINT
    """
    sys.exit()


class BasePath(object):
    def __init__(self, root_dir=" ", affine_transformation=" ",
                 perspective_transfrom=" ", hour_min=0.0, hour_max=24.0):
        """
        Args:
            [root_dir(string)]: Location of the root directory where images are
            [affine_transformation(str)]: Path to file containing
                affine_tranformation
            [hour_min(float)]: Minimium hour to consider images. Images which
                are below this amount will be discarded
            [hour_max(float)]: Maximium hour to consider images. Images which
                are above this amount will be discarded
        """

        # Dictionary to transform dates from motnh to number
        self.dates = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'April': '04',
                      'May': '05', 'June': '06', 'July': '07', 'Aug': '08',
                      'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

        # Find physical location of root directory
        if root_dir == " ":
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir

        # Specify bounds on system by hour
        self.hour_min = hour_min
        self.hour_max = hour_max

        # Point to Triggers.txt file
        trigger_path = self.root_dir + "/Triggers.txt"
        trigger_file_present = os.path.isfile(trigger_path)

        # Check if file exits. If so, load data
        if trigger_file_present:
            # Load all trigger dates
            #self.trigger_dates = self._trigger_files(trigger_path)
            self.trigger_dates = []
        else:
            self.trigger_dates = []

        # Find all subdirectory folders under root directory
        self.sub_directories = self._subdirs()

        # Handle logging
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def get_hour(self, name):
        """
        Convert from folder name type to hour in 24-hour format

        Input:
            name (str): Input in form YYYY_MM_DD hh_mm_ss
        Return:
            Hour (int): Hour in 0-24 hour form
        """
        full_time = name.split(" ")
        hour = full_time[1].split("_")[0]

        return int(hour)

    def _subdirs(self):
        """
        Returns list of subdirectories under main directory which satisifes
        the time constraint

        Input:
            None
        Return:
            return_subdirs(list<str>): List of all subdirectories which satisfy
                                       input paramaters
        """
        # Get list of all folders in the current directory
        all_subdirs = get_paths(self.root_dir + "/*/")

        # Only return sub directories that contain images, i.e. begin with 2018
        return_subdirs = []
        for path in all_subdirs:
            names = path.split('/')
            # all folders start with 20. If a folder starts with 2018, append
            # to return list
            folder_name = names[len(names) - 2]

            if folder_name[0:2] == "20":
                #print(names[len(names) - 2])
                hour = self.get_hour(names[len(names) - 2])
                print(hour)
                if hour >= self.hour_min and hour <= self.hour_max:

                    return_subdirs.append(path)
        return return_subdirs

    def _trigger_files(self, path):
        '''
        Determine all 'trigger events' from file Trigger.txt

        Input:
            path(str): Path which poitns to Trigger.txt file
        Return:
            None
        '''
        trigger_dates = set()
        # Open path location
        with open(path, "r") as f:
            for line in f:
                '''
                File is listed out of order and in a different format than
                the file systems where the events actually happen. Must
                transform
                '''
                day = (line.split(" ")[3]).split(
                    "-")  # Get the year, month, and date
                time_stamp = (line.split(" ")[4]).split(
                    "-")[0].replace(':', '_')[:-1]  # Hour, min, and second
                # Determine the month number from month (e.g. 01 for Jan)
                month = self.dates[day[1]]
                # Transform to folder date form, YYYY-MM-DD
                new_date = day[2] + "_" + month + \
                    "_" + day[0] + "_" + time_stamp
                # Append to list and return
                trigger_dates.add(new_date)


class AMP3GImageProc():
    """
    Class containign modules to help process 3G-AMP image data

    Attributes:
        -homog (np.mat): Homography transformation matrix between images
        -time_delay_allowed: Maximium allowable time between images timestamps
        -save_directory(str): Location to save events

    Methods:
        -image_overlap: Check overlap between stereo images
        -background_subtraction: Runs openCv createBackgroundSubtractorMOG2 alg.
            for all subdirectories under the root directory
        -single_directiory_background_subtraction: Runs openCv
            createBackgroundSubtractorMOG2 alg. for one subdirectory
        -get_hour: Determines hour from full image/folder name
    """

    def __init__(self,  save_directory=' ', time_delay_allowed=0.1,
                 homography_transform=' '):
        """
        Class containign modules to help process 3G-AMP image data

        Input:
            -[save_directory(str)]: Location to save events
            -[time_delay_allowed(float)]: Maximium allowable time between images timestamps
            -[homography_transform(string)]: Location of homography transform file

        Return:
            None
        """
        if homography_transform == " ":
            self.homog = np.identity(3)

        else:
            try:
                file = open(homography_transform, "r")
                self.homog = np.array(file.read().split(',')[0:9],
                                      dtype=np.float32).reshape((3, 3))
            except:
                print("Homography file not found")
                sys.exit(0)

        self.time_delay_allowed = time_delay_allowed
        self.save_directory = save_directory
        self.overlap_sum_threshold = 16423336  # 2000000 #Threshold for overlap

    def display_images(self, path):
        """
        Display all images in a directory

        Input:
            path(str): Base location of Manta 1 and Manta 2 folders

        Return:
            None
        """
        cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame1', 1200, 1200)

        self._display_images(path + "/Manta 1/*.jpg",
                             path + "/Manta 2/*.jpg")

    def image_overlap(self, path, display_images=False, display_overlap=False, save=True):
        """
        Check the overlap of images, check image intensity
        Inputs:
            path(str): Base location of Manta 1 and Manta 2 folders
            [display_images(bool)]: Display the images
            [display_overlap(bool)]: Display the overlap between transformed
                images
        Returns:
            -overlap_intensity(list<float>): List of image overlaps from the
                defined location
        """

        overlap_intensity = self._overlap(path + "Manta 1/*.jpg",
                                          path + "Manta 2/*.jpg", overlap=True,
                                          display_images=display_images,
                                          display_overlap=display_overlap,
                                          color=True, date=None, save=save)
        # print(overlap_intensity)
        return overlap_intensity

    def _display_images(self, d1, d2, time_delay=1):
        """
        Display all jpgs under d1 and d2 path
        Inputs:
            -d1(str): Points to Manta1 images
            -d1(str): Points to Manta2 images
            -[time_delay(int)]: Time (in ms) to display image for
        Return:
            None
        """
        images1 = sorted(get_paths(d1), reverse=True)
        images2 = sorted(get_paths(d2), reverse=False)

        images = zip(images1, images2)

        for fname1, fname2 in images:

            #signal.signal(signal.SIGINT, sigint_handler)
            img1 = cv2.imread(fname1)
            cv2.imshow('frame1', img1)
            k = cv2.waitKey(time_delay)
            if k == 99:
                cv2.destroyAllWindows()
                sys.exit()

    def _overlap(self, d1, d2, overlap=False,
                 display_images=False, color=False,
                 display_overlap=False,
                 date=None, save=True):
        """
        Run background subtraction algorithm using openCv
        createBackgroundSubtractorMOG2. Will do background subtraction for all
        images in d1 and d2 (directories 1 and 2, respectively). d1 and d2
        inputs should be the directories where each of the stereo camera iamges
        are located.

        The function will also claculate overlap, if desired, and return the
        intensity of the image(s) overlap. Image one's frame will be transformed
        into image one's frame and will check for overlap. If false, return
        empty list

        Input:
            d1(str): Directory 1 containing images (i.e subdir + Manta 1)
            d2(str): Directory 2 containing images
            [overlap(bool)]: Calcuate image overlap
            [display_images(bool)]: Display images
            [color(bool)]: Display color or grayscale images
            [display_overlap(bool)]: Display overlapped image
            [date(str)]: Current date (YYYY_MM_DD HH_MM_SS) for saving
            [save(bool)]:To save or not to save

        Return:
            overlap_intensity(list<float>): List of all image intensities
        """
        # get list of all images from current directory
        images1 = sorted(get_paths(d1), reverse=True)
        images2 = sorted(get_paths(d2), reverse=True)

        # create a background subtraction object
        fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # zip images so that we can loop through image names
        images = zip(images1, images2)

        # initialize window size
        if display_images:
            cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame1', 1200, 1200)
            cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame2', 1200, 1200)
        if display_overlap:
            cv2.namedWindow('overlap', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('overlap', 800, 800)

        # Create list of image_intensity
        overlap_intensity = []

        # Kernel for median blur
        kernel = np.ones((250, 250), np.uint8)
        overlap_count = 0
        i = 0
        """
        Log file
        """
        with open(self.save_directory + '\log.txt', 'a+') as f:
            now = datetime.datetime.now()
            f.write("%s, %s\n" %
                    (d1.split('/')[-3], now.strftime("%Y_%m_%d %H::%M::%S")))

        for fname1, fname2 in images:

            """
            Loop through all image frames and run background subtraction.
            If overlap is selected, compare the overlap between the two
            images
            """
            signal.signal(signal.SIGINT, sigint_handler)

            check_date = self._check_date(fname1, fname2)
            if not check_date:
                fname1, fname2 = self._find_date(images, i)
                if fname1 == None:  # No images within allowable time
                    check_date = False
                else:
                    check_date = True
            if check_date:
                # Read images and inport
                img1 = cv2.imread(fname1)
                img2 = cv2.imread(fname2)

                # Apply the mask
                img1b = fgbg1.apply(img1)  # , learningRate=0.035)
                img2b = fgbg2.apply(img2)  # , learningRate=0.035)

                ret, thresh1 = cv2.threshold(
                    img1b, 125, 255, cv2.THRESH_BINARY)
                ret, thresh2 = cv2.threshold(
                    img2b, 125, 255, cv2.THRESH_BINARY)

                # Apply a median blur to reduce noise
                blur1 = cv2.medianBlur(thresh1, 25)
                blur2 = cv2.medianBlur(thresh2, 25)

                if overlap:

                    blur1_trans = cv2.warpPerspective(blur2, self.homog,
                                                      (blur1.shape[1], blur1.shape[0]))

                    blur1_trans_dilate = cv2.dilate(
                        blur1_trans, kernel, iterations=1)
                    blur2_dilate = cv2.dilate(blur1, kernel, iterations=1)

                    # Check Overlap between images using bitwise_and
                    overlap_img = np.bitwise_and(
                        blur2_dilate, blur1_trans_dilate)
                    overlap_sum = np.sum(overlap_img)
                    #print(overlap_sum, overlap_sum>self.overlap_sum_threshold)
                    if overlap_sum > self.overlap_sum_threshold:  # IT ALWAYS FAILS ON THE FIRST TRY
                        overlap_count += 1
                    else:
                        overlap_count = 0
                    if 1:  # overlap_count >= 4 and save:
                        print("HIGH OVERLAP DETECTED FOR DATE:",
                              d1.split('/')[-3])
                        with open(self.save_directory + '/highStereoData_Q.txt', 'a+') as f:
                            f.write("%s, %s, %s, %s\n" % (d1.split('/')[-3], '/'.join(fname1.split(
                                '/')[-2:]), '/'.join(fname2.split('/')[-2:]), now.strftime("%Y_%m_%d %H::%M::%S")))
                        break

                    overlap_intensity.append(overlap_sum)
                    if display_overlap:
                        cv2.imshow('overlap', overlap_img)

                if display_images:
                    if color:

                        cv2.imshow('frame1', img1)
                        cv2.imshow('frame2', img2)
                    else:
                        cv2.imshow('frame1', blur1)
                        cv2.imshow('frame2', blur2)

                if display_images or display_overlap:
                    if i == 0:
                        k = cv2.waitKey(100)
                    else:
                        k = cv2.waitKey(100)

                    if k == 99:
                        cv2.destroyAllWindows()
                        sys.exit()
            i += 1
            # print(i)

            # Return list of overlap intensities or empty list
        return overlap_intensity

    def _check_date(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name

        Return:
            Bool: If timestamps are close enough together


        """
        # return True
        time1 = float(
            '.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float(
            '.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))

        if abs(time1 - time2) < self.time_delay_allowed:
            return True

        return False

    def _time_diff(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name

        Return:
            Bool: If timestamps are close enough together


        """
        time1 = float(
            '.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float(
            '.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))

        return abs(time1 - time2)

    def _check_day_hour(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name

        Return:
            Bool: If timestamps are close enough together


        """
        day1 = f1.split('/')[-1].split('_')
        day1 = '_'.join(day1[:6])
        day2 = f2.split('/')[-1].split('_')
        day2 = '_'.join(day2[:6])
        if day1 == day2:
            return True

        return False

    def _find_date(self, images, i):
        fname1 = images[i][0]
        fname2 = images[0][1]
        prev_time_diff = self._time_diff(fname1, fname2)

        for loc in range(0, len(images)):
            if self._check_day_hour(fname1, images[loc][1]):
                if self._check_date(fname1, images[loc][1]):

                    return fname1, images[loc][1]
                #print(fname1, images[loc][1], self._time_diff(fname1, images[loc][1]) > prev_time_diff)
                if self._time_diff(fname1, images[loc][1]) > prev_time_diff:
                    # Diverging, break
                    return None, None  # fname1, images[loc-1][1]
                prev_time_diff = self._time_diff(fname1, images[loc][1])
        return None, None
