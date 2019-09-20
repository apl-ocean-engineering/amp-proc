#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""

import signal
import datetime

import cv2
import numpy as np


class imageTransforms(object):
    """
    Class to help determine transformation between frames in two 3G-AMP cameras

    Attributes:
        -Images_path(str): Path to directory containing images for calibration
        -x#_points, y#_points (list<float>): 4 lists containing corresponding
            points in each camera frame
        -image1, image2 (np.mat<float>): Images
        -m1_subdirectories, m2_subdirectories (list<str>): List
            containing image paths
    Methods:
        -corresponding_image_points: Manual correspondance of points between
            two image frames
        -find_perspective: Calculates the perspective transform matrix
        -find_homography: Calculates the homography transform matrix
        -find_affine: Calculates the affine transform matrix
        -get_points: Returns corresponding image points between the two frames

    """

    def __init__(self, images_path, images_path2=" "):
        """
        Args:
            images_path(str): Path pointing to location of images
        """

        self.images_path = images_path
        self.images_path2 = images_path2
        self.x1_points = []
        self.y1_points = []
        self.x2_points = []
        self.y2_points = []
        self.image1 = np.zeros([0, 0])
        self.image2 = np.zeros([0, 0])

        self.m1_subdirectories, self.m2_subdirectories, self.m1_subdirectories2, self.m2_subdirectories2 = self._subdirs()

        self.time_delay_allowed = 0.05

    def corresponding_image_points(self):
        """
        Determine coressponding image points between the frames

        Will display two WAMP images. User must click on identical point
        in two frames. x#_points, and y#_points will populate as the user
        clicks on points
        """
        # Initalzie image windows
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image1', 1200, 1200)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image2', 1200, 1200)
        # Define mouse callback functions
        cv2.setMouseCallback('image1', self._mouse_click1)
        cv2.setMouseCallback('image2', self._mouse_click2)
        # print(self.m1_subdirectories)
        # Loop through all images in subdirectory location
        print("Click on the same point in both images")
        print("Press enter to move to next corresponding images")
        print("Press 'f' to finish")
        print("Press cntrl+c to quit")
        for i in range(0, len(self.m1_subdirectories)):
            signal.signal(signal.SIGINT, self._sigint_handler)
            # Get img1 and img2 from the subdirectories
            f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]

            self.img1, self.img2 = cv2.imread(f1), cv2.imread(f2)
            # Show images
            cv2.imshow('image1', self.img1)
            cv2.imshow('image2', self.img2)
            # Press 'enter' to move on, f to finish, cntrl+c to quit
            k = cv2.waitKey(0)
            if k == 99:
                cv2.destroyAllWindows()
                sys.exit()
            if k == 102:
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()

    def find_perspective(self, save=False, path=""):
        """
        Calculate perpsective transformation matrix from corresponding points

        NOTE: Must be exactly 4 points, or error will raise

        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired

        Return:
            perspective_transform (np.mat<float>): (3X3) transformation matrix
        """

        # Get corresponding points
        pnts1, pnts2 = self._image_points()
        if len(pnts1) != 4 or len(pnts2) != 4:
            raise ValueError(
                "Must have exactly four corresponding image points")
        # Get transform
        perspective_transform = cv2.getPerspectiveTransform(pnts1, pnts2)
        if save:
            # Save data to text file
            np.savetxt(path + "perspective_transform.txt",
                       perspective_transform.reshape(1, 9),
                       delimiter=',', fmt="%f")

        return perspective_transform

    def find_homography(self, save=False, path=""):
        """
        Calculate homography transformation matrix from corresponding points

        Should use at least four points to be accruate

        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired

        Return:
            homography_transform (np.mat<float>): (2X3) homography matrix

        #Get corresponding points
        pnts1, pnts2 = self._image_points()
        #Get transform
        homography_transform = cv2.findHomography(pnts1, pnts2)

        if save:
            #Save data to text file
            np.savetxt(path+"3Ghomography_transform.txt",
                       np.array(homography_transform[0]).reshape(1,9),
                       delimiter=',', fmt="%f")

        return homography_transform

        """
        images = zip(self.m1_subdirectories, self.m2_subdirectories)
        for i in range(0, len(self.m1_subdirectories)):
            # print(i)
            try:
                signal.signal(signal.SIGINT, sigint_handler)
                # Get img1 and img2 from the subdirectories
                f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]

                check_date = self._check_date(f1, f2)

                if not check_date:
                    f1, f2 = self._find_date(images, i)
                    if f1 == None:  # No images within allowable time
                        check_date = False
                    else:
                        check_date = True
                if check_date:
                    print(f1, f2)
                    img1, img2 = cv2.imread(f1), cv2.imread(f2)

                    orb = cv2.ORB_create()
                    kp1, des1 = orb.detectAndCompute(
                        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
                    kp2, des2 = orb.detectAndCompute(
                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                    matches = bf.match(des1, des2)
                    for mat in matches:

                        self.x1_points.append(kp1[mat.queryIdx].pt[0])
                        self.y1_points.append(kp1[mat.queryIdx].pt[1])
                        self.x2_points.append(kp2[mat.trainIdx].pt[0])
                        self.y2_points.append(kp2[mat.trainIdx].pt[1])

            except:
                pass
        pnts1, pnts2 = self.get_points()

        fundmental = cv2.findFundamentalMat(pnts1, pnts2)

        print(fundmental)

        ret, H1, H2 = cv2.stereoRectifyUncalibrated(
            pnts1, pnts2, fundmental[0], (img2.shape[1], img2.shape[0]))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 800, 800)
        cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('overlay', 800, 800)
        """
        cv2.namedWindow('overlay1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('overlay1', 800,800)
        cv2.namedWindow('overlay2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('overlay2', 800,800)
        """

        for i in range(0, len(self.m1_subdirectories)):
            f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]

            check_date = self._check_date(f1, f2)

            if not check_date:
                f1, f2 = self._find_date(images, i)
                if f1 == None:  # No images within allowable time
                    check_date = False
                else:
                    check_date = True
            if check_date:

                img1 = cv2.imread(f1)
                img2 = cv2.imread(f2)
                dst1 = cv2.warpPerspective(
                    img1, H1, (img1.shape[1], img1.shape[0]))
                dst2 = cv2.warpPerspective(
                    img2, H2, (img2.shape[1], img2.shape[0]))

                overlay1 = cv2.addWeighted(dst1, 1.0, dst2, 1.0, 0.0)

                cv2.imshow("img", img1)
                print(f1, f2)
                cv2.imshow("overlay", overlay1)

                cv2.waitKey(100)

    def find_homography_check(self, save=False, path=""):
        """
        Calculate homography transformation matrix from corresponding points

        Should use at least four points to be accruate

        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired

        Return:
            homography_transform (np.mat<float>): (2X3) homography matrix

        #Get corresponding points
        pnts1, pnts2 = self._image_points()
        #Get transform
        homography_transform = cv2.findHomography(pnts1, pnts2)

        if save:
            #Save data to text file
            np.savetxt(path+"3Ghomography_transform.txt",
                       np.array(homography_transform[0]).reshape(1,9),
                       delimiter=',', fmt="%f")

        return homography_transform
        """

        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        images = zip(self.m1_subdirectories, self.m2_subdirectories)

        # print(self.m1_subdirectories2)
        images = zip(self.m1_subdirectories2, self.m2_subdirectories2)
        for i in range(0, min(len(self.m1_subdirectories2), 100)):
            # print(i)
            if 1:  # i % 2 == 0:
                try:
                    signal.signal(signal.SIGINT, sigint_handler)
                    # Get img1 and img2 from the subdirectories
                    f1, f2 = self.m1_subdirectories2[i], self.m2_subdirectories2[i]

                    check_date = self._check_date(f1, f2)

                    if not check_date:
                        f1, f2 = self._find_date(images, i)
                        if f1 == None:  # No images within allowable time
                            check_date = False
                        else:
                            check_date = True

                    if check_date:
                        #print(2, f1, f2)
                        img1, img2 = cv2.imread(f1), cv2.imread(f2)

                        orb = cv2.ORB_create()
                        kp1, des1 = orb.detectAndCompute(
                            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
                        kp2, des2 = orb.detectAndCompute(
                            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                        matches = bf.match(des1, des2)
                        print(len(matches))
                        cv2.imshow('image1', img1)
                        cv2.imshow('image2', img2)
                        cv2.waitKey(100)

                        for mat in matches:

                            self.x1_points.append(kp1[mat.queryIdx].pt[0])
                            self.y1_points.append(kp1[mat.queryIdx].pt[1])
                            self.x2_points.append(kp2[mat.trainIdx].pt[0])
                            self.y2_points.append(kp2[mat.trainIdx].pt[1])

                            img3 = cv2.drawMatches(
                                img1, kp1, img2, kp2, matches[:10], img1, flags=2)
                            cv2.imshow('image3', img3)

                            k = cv2.waitKey(1)
                            if k == 99:
                                cv2.destroyAllWindows()
                                sys.exit()
                            if k == 102:
                                cv2.destroyAllWindows()
                                break

                except:
                    pass

    def stereo_rectify(self, path="", save=False):
        intrinsics1 = path + "camera1/intrinsic_matrix.csv"
        file = open(intrinsics1, "r")
        K1 = np.array((file.read().replace('\n', ',')).split(',')[0:9],
                      dtype=np.float64).reshape((3, 3))
        intrinsics2 = path + "camera2/intrinsic_matrix.csv"
        file = open(intrinsics2, "r")
        K2 = np.array((file.read().replace('\n', ',')).split(',')[0:9],
                      dtype=np.float64).reshape((3, 3))
        distortion1 = path + "camera1/distortion_coeffs.csv"
        file = open(distortion1, "r")
        d1 = np.array((file.read().replace('\n', ',')).split(',')[0:5],
                      dtype=np.float64).reshape((1, 5))[0]
        distortion2 = path + "camera2/distortion_coeffs.csv"
        file = open(distortion2, "r")
        d2 = np.array((file.read().replace('\n', ',')).split(',')[0:5],
                      dtype=np.float64).reshape((1, 5))[0]
        imsize = (2056, 2464)
        rotation = path + "rotation_matrix.csv"
        file = open(rotation, "r")
        R = np.array((file.read().replace('\n', ',')).split(',')[0:9],
                     dtype=np.float64).reshape((3, 3))
        translation = path + "translation_matrix.csv"
        file = open(translation, "r")
        T = np.array((file.read().replace('\n', ',')).split(',')[0:3],
                     dtype=np.float64).reshape((1, 3))[0]

        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K1, d1, K2, d2,  imsize,
                                                    R, T, alpha=-1)

        cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img1', 800, 800)
        cv2.namedWindow('blur1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blur1', 800, 800)
        cv2.namedWindow('blur2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blur2', 800, 800)

        fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        kernel = np.ones((15, 15), np.uint8)
        images = zip(self.m1_subdirectories, self.m2_subdirectories)
        for i in range(0, min(len(self.m1_subdirectories), 100)):
            signal.signal(signal.SIGINT, sigint_handler)
            # Get img1 and img2 from the subdirectories
            f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]

            check_date = self._check_date(f1, f2)

            if not check_date:
                f1, f2 = self._find_date(images, i)
                if f1 == None:  # No images within allowable time
                    check_date = False
                else:
                    check_date = True

            if check_date:
                img1, img2 = cv2.imread(f1), cv2.imread(f2)
                img1 = cv2.imread(f1)
                img2 = cv2.imread(f2)

                img1b = fgbg1.apply(img1)
                img2b = fgbg2.apply(img2)

                stereo = cv2.StereoBM_create(numDisparities=320, blockSize=5)
                disparity = stereo.compute(img1b, img2b)
                cv2.imshow("img1", img1b)
                cv2.imshow("blur1", disparity)

                k = cv2.waitKey(100)
                if k != -1:
                    sys.exit(1)

    def find_affine(self, save=False, path=""):
        """
        Calculate affine transformation matrix from corresponding points

        NOTE: Must be exactly 3 points, or error will raise

        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired

        Return:
            homography_transform (np.mat<float>): (2X3) homography matrix
        """
        # Get corresponding points
        pnts1, pnts2 = self._image_points()
        if len(pnts1) != 3 or len(pnts2) != 3:
            raise ValueError(
                "Must have exactly three corresponding image points")
        # Get affine transform
        affine_transform = cv2.getAffineTransform(pnts1, pnts2)

        if save:
            # Save data to text file
            np.savetxt(path + "affine_transform.txt",
                       affine_transform.reshape(1, 6), delimiter=',',
                       fmt="%f")

        return affine_transform

    def get_points(self):
        """
        Return corresponding image points

        Return:
            points1, points2 (list<tuple<float>>): Corresponding points
        """
        points1, points2 = self._image_points()

        return points1, points2

    def _image_points(self):
        """
        Organize image points into two lists of corresponding tuples

        Return:
            pnts1, pnts2 (list<tuple<float>>): Corresponding points
        """
        # Check that points clicked are equal
        if len(self.x1_points) != len(self.x2_points):
            raise AttributeError("Unequal Points Clicked")
        # Organize points
        pnts1 = []
        pnts2 = []
        points1 = []
        points2 = []
        for i in range(0, len(self.x1_points)):
            pnts1.append(np.array([[np.float32(self.x1_points[i]),
                                    np.float32(self.y1_points[i])]]))
            pnts2.append(np.array([[np.float32(self.x2_points[i]),
                                    np.float32(self.y2_points[i])]]))

        # Must be float 32s to work in OpenCV
        pnts1 = np.float32(pnts1)
        pnts2 = np.float32(pnts2)

        return pnts1, pnts2

    def _mouse_click1(self, event, x, y, flags, param):
        """
        Callback function for mouse click event on image1 frame

        Places clicked points into x1_ and y1_points lists
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1_points.append(x)
            self.y1_points.append(y)
            # Draw circle where clicked
            cv2.circle(self.img1, (x, y), 20, (255, 0, 0), -1)
            cv2.imshow('image1', self.img1)

    def _mouse_click2(self, event, x, y, flags, param):
        """
        Callback function for mouse click event on image2 frame

        Places clicked points into x2_ and y2_points lists
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x2_points.append(x)
            self.y2_points.append(y)
            # Draw circle where clicked
            cv2.circle(self.img2, (x, y), 20, (255, 0, 0), -1)
            cv2.imshow('image2', self.img2)

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
