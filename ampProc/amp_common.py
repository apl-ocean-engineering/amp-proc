#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu

Common scripts amongst AMP processing code
"""

import datetime
import glob
import sys
import time
import numpy as np


def save_np_array(save_name, mat):
    np.savetxt(save_name, mat, fmt="%1.3f", delimiter=",")


class CvSquare:
    """
    Simple square object which will 'move', for AMP rectified image stereo
    fish detection
    """
    def __init__(self, img_size):
        """
        Input:
            img_size(tuple (x,y)): Size of image to verify in bounds status
        """
        self.lower_x = 0
        self.lower_y = 0
        self.upper_x = 0
        self.upper_y = 0

        self.img_size = img_size

    def set_square(self, lower_x, lower_y, upper_x, upper_y):
        """
        Define square values
        """
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.upper_y = upper_y

    def set_from_YOLO_square(self, sq):
        """
        Set from YOLO detection square
        """
        self.lower_x = sq.lower_x
        self.lower_y = sq.lower_y
        self.upper_x = sq.upper_x
        self.upper_y = sq.upper_y

    def move_square(self, motion=1):
        """
        Move square in forward horizontal direction by motion (default 1)
        """
        if motion > 0:
            if (self.upper_x < self.img_size[0]):
                self.lower_x += motion
                self.upper_x += motion
                return True
        elif motion < 0:
            if (self.upper_x > 0):
                self.lower_x += motion
                self.upper_x += motion
                return True

        return False

    def extend_square(self, x_extend, y_extend):
        """
        Increase a square by (x,y) in both directions
        """
        self.lower_x = max(0, self.lower_x - x_extend)
        self.lower_y = max(0, self.lower_y - y_extend)
        self.upper_x = min(self.img_size[0], self.upper_x + x_extend)
        self.upper_y = min(self.img_size[1], self.upper_y + y_extend)


class ampCommon:
    def __init__(self, time_delay_allowed=0.05):
        self.time_delay_allowed = time_delay_allowed

    def relative_time_diff(self, f1, f2):
        """
        Verify that image timestamps are less than time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name

        Return:
            Float: Timestamps differance
        """
        minute1 = f1.split('/')[-1].split('_')[-2]
        minute2 = f2.split('/')[-1].split('_')[-2]
        hour1 = f1.split('/')[-1].split('_')[-3]
        hour2 = f2.split('/')[-1].split('_')[-3]
        #print(hour1, hour2)
        time1 = float(
            '.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float(
            '.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))
        diff = time1 - time2
        hour_adjust = False
        if hour1 > hour2:
            diff+=60
            hour_adjust = True
        elif hour1 < hour2:
            hour_adjust = True
            diff-=60
        if not hour_adjust:
            if minute1 > minute2:
                diff+=60
            elif minute1 < minute2:
                diff-=60
        #print(hour1 < hour2, diff, time1, time2)
        return diff

    def time_diff(self, f1, f2):
        """
        Verify that image timestamps are less than time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name

        Return:
            Float: Timestamps differance
        """
        time1 = float(
            '.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float(
            '.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))

        return abs(time1 - time2)

    def find_date(self, images, i):
        fname1 = images[i][0]
        fname2 = images[0][1]
        #print(len(images), i)


        #print(fname1, fname2)
        prevtime_diff = self.time_diff(fname1, fname2)
        #print(prevtime_diff)

        for loc in range(0, len(images)):
            #print(fname1, images[loc][1])
            #print(self._check_day_hour(fname1, images[loc][1]))
            if self._check_day_hour(fname1, images[loc][1]):
                #if fname1 == '/home/mitchell/WAMP_workspace/WETS_stereo_test_images/2018_12_14/2018_12_14 16_29_11/Manta 1/2018_12_14_16_28_11.38.jpg':
                #print(fname1)
                #print(images[loc][1])
                #print(self.relative_time_diff(fname1, images[loc][1]))
                #print(self._check_date(fname1, images[loc][1]))
                if self._check_date(fname1, images[loc][1]):
                    return fname1, images[loc][1]
                if self.time_diff(fname1, images[loc][1]) > prevtime_diff:
                    #print(fname1, images[loc][1])
                    #print(prevtime_diff)
                    #print(self.time_diff(fname1, images[loc][1]))
                    # Diverging, break
                    return None, None  # fname1, images[loc-1][1]
                prevtime_diff = self.time_diff(fname1, images[loc][1])
        return None, None

    def beyond_date(self, date, start_date):
        if start_date == ' ':
            return True
        else:
            year = int(date.split('_')[0])
            month = int(date.split('_')[1])
            day = int(date.split('_')[2])
            date1 = datetime.date(year=year, month=month, day=day)
            start_date = start_date.split("/")[3]
            start_year = int(start_date.split('_')[0])
            start_month = int(start_date.split('_')[1])
            start_day = int(start_date.split('_')[2])
            date2 = datetime.date(
                year=start_year, month=start_month, day=start_day)
            if time.mktime(date1.timetuple()) > time.mktime(date2.timetuple()):
                return True
            return False

    def find_img_sonar(self, manta_folders, sonar_folders, img_count, sonar_count):
        manta_fname = manta_folders[img_count]
        sonar_fname = sonar_folders[sonar_count]

        time_diff = self.relative_time_diff(manta_fname, sonar_fname)
        while not time_diff < self.time_delay_allowed:
            if img_count >= len(manta_folders) or sonar_count >= len(sonar_folders):
                return " ", " "
            manta_fname = manta_folders[img_count]
            sonar_fname = sonar_folders[sonar_count]
            time_diff = self.relative_time_diff(manta_fname, sonar_fname)
            if time_diff < 0:
                img_count += 1
            elif time_diff > 0:
                sonar_count += 1

        return manta_fname, sonar_fname

    def find_image_near_sonar(self, sonar_fname, manta_imgs, diff_allowed = 1.0):
        manta_fname = sonar_fname.replace('BlueView', 'Manta 1').rstrip()
        #print(manta_fname, manta_fname in manta_imgs)
        if manta_fname in manta_imgs:
            return manta_imgs

        time_diff = np.inf
        return_matna_fname = ' '
        for manta_fname in manta_imgs:
            new_time_diff = abs(self.relative_time_diff(manta_fname, sonar_fname))
            if new_time_diff < time_diff:
                #print(new_time_diff)
                if new_time_diff < diff_allowed:
                    return_matna_fname = manta_fname
                time_diff = new_time_diff

        return return_matna_fname
    #
    # def find_all_images_near_sonar(self, manta_folders, sonar_folders, sonar_count):
    #
    #         manta_fname = manta_folders[0]
    #         sonar_fname = sonar_folders[sonar_count]
    #         time_diff = self.relative_time_diff(manta_fname, sonar_fname)
    #         if time_diff < self.time_delay_allowed:
    #             return sonar_fname, [manta_fname]
    #         manta_return_folders = []
    #         img_count = 1
    #
    #         #print(img_count, len(manta_folders))
    #         while img_count < len(manta_folders):
    #             manta_fname = manta_folders[img_count]
    #
    #             time_diff = self.relative_time_diff(manta_fname, sonar_fname)
    #             #print(time_diff)
    #             if time_diff < self.time_delay_allowed:
    #                 manta_return_folders.append(manta_fname)
    #
    #             img_count += 1
    #
    #         return sonar_fname, manta_return_folders

    def _subdirs(self):
        """
        Return list of all subdirectories under current directory containing
        the Manta 1 and Manta 2 images

        Return:
            -manta1_subdirs, manta2_subdirs (list<str>): Paths for all images
        """
        # Get list of all folders in the current directory
        manta1_subdirs = sorted(
            glob.glob(self.images_path + "/Manta 1/*.jpg"), reverse=True)
        manta2_subdirs = sorted(
            glob.glob(self.images_path + "/Manta 2/*.jpg"), reverse=True)
        manta1_subdirs2 = None
        manta1_subdirs2 = None
        if self.images_path2 != " ":
            manta1_subdirs2 = sorted(
                glob.glob(self.images_path2 + "/Manta 1/*.jpg"), reverse=True)
            manta1_subdirs2 = sorted(
                glob.glob(self.images_path2 + "/Manta 2/*.jpg"), reverse=True)

        return manta1_subdirs, manta2_subdirs, manta1_subdirs2, manta1_subdirs2

    def _sigint_handler(self, signum, frame):
        """
        Exit system if SIGINT
        """
        sys.exit()

    def _check_date(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed
        apart
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
        #print(f1.split('/')[-1], f2.split('/')[-1])
        #print(time1, time2)
        if abs(time1 - time2) < self.time_delay_allowed:
            return True

        return False

    def _check_day_hour(self, f1, f2):
        """
        Verify that image timestamps are less than time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name

        Return:
            Bool: If timestamps are close enough together
        """
        day1 = f1.split('/')[-1].split('_')
        #print(day1)
        day1 = '_'.join(day1[:4])
        day2 = f2.split('/')[-1].split('_')
        day2 = '_'.join(day2[:4])

        #print(day1, day2)


        if day1 == day2:
            return True

        return False
