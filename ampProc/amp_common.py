#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""

import signal
import datetime
import glob
import sys
import time
import cv2
import numpy as np

class ampCommon:
    def __init__(self, time_delay_allowed=0.05):
        self.time_delay_allowed = time_delay_allowed

    def time_diff(self, f1, f2):
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

    def find_date(self, images, i):
        fname1 = images[i][0]
        fname2 = images[0][1]
        prevtime_diff = self.time_diff(fname1, fname2)

        for loc in range(0, len(images)):
            if self._check_day_hour(fname1, images[loc][1]):
                if self._check_date(fname1, images[loc][1]):

                    return fname1, images[loc][1]
                if self.time_diff(fname1, images[loc][1]) > prevtime_diff:
                    # Diverging, break
                    return None, None  # fname1, images[loc-1][1]
                prevtime_diff = self.time_diff(fname1, images[loc][1])
        return None, None

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

        if abs(time1 - time2) < self.time_delay_allowed:
            return True

        return False

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

    def _beyond_date(self, date, start_date):
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
