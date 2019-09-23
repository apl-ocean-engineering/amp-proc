#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import numpy as np
import cv2


class Loader:
    """
    Loader class to pass to Extrinsic/Intrinsic loader
    Can either set base path with all paths relative to that path, or
    do all relative paths
    Form:
      Paramter = [load (bool), path (string)]
      Pass path with '/' appended to front for full path. Else, relative path
    """
    base_path = " "
    # Intrnsic paramaters and distortion
    K1 = [False, " "]
    K2 = [False, " "]
    d1 = [False, " "]
    d2 = [False, " "]
    # Rotation and Translation
    R = [False, " "]
    t = [False, " "]
    # Projection matrices
    P1 = [False, " "]
    P2 = [False, " "]

    def __init__(self):
        """
        Arg calibration loader: Loaded calibration.yaml file
        """
        self.parms = dict()

    def load_params_from_file(self, calibration_loader):
        self.parms["base_path"] = self.base_path
        self.parms["K1"] = self.K1
        self.parms["K2"] = self.K2
        self.parms["d1"] = self.d1
        self.parms["d2"] = self.d2
        self.parms["R"] = self.R
        self.parms["t"] = self.t
        self.parms["P1"] = self.P1
        self.parms["P2"] = self.P2
        for key in calibration_loader.keys():
            self.parms[key] = calibration_loader[key]
        self.set_params()

    def set_params(self):
        self.base_path = self.parms["base_path"]
        self.K1 = self.parms["K1"]
        self.K2 = self.parms["K2"]
        self.d1 = self.parms["d1"]
        self.d2 = self.parms["d2"]
        self.R = self.parms["R"]
        self.t = self.parms["t"]
        self.P1 = self.parms["P1"]
        self.P2 = self.parms["P2"]


class Paramters:
    """
    Paramaters class for extrinsic and intrinsic stereo camera properties
    K1/2: Intrinsic Paramaters
    d1/d2: Distortion paramaters
    R: Rotation matrix between cameras
    t: translation vector between cameras
    P1/P2: Projection matrix
    """
    im_size = (0, 0)
    K1 = np.eye(3)
    K2 = np.eye(3)
    d1 = np.zeros(5)
    d2 = np.zeros(5)
    R = np.eye(3)
    t = np.zeros(3)
    P1 = None
    P2 = None


class ExtrinsicIntrnsicLoaderSaver:
    """
    Helper class to load and calculate projection matricies
    Attributes:
        paramters: Object containing camera paramaters
    Methods:
        calculate_projection_matracies: Calculate projection matrices and update
        paramters
    """
    def __init__(self, paramLoader, im_size):
        """
        Input:
            paramLoader: Dictonary of calibration matricies
        """
        self.paramaters = Paramters
        self.paramaters.im_size = im_size
        self._load_params(paramLoader)

    def calculate_projection_matracies(self):
        """
        Calculates projection matricies from R and t matricies. Updates
        paramters attribute
        """
        _p1 = np.zeros((3, 4), dtype=float)
        _p1[0, 0] = 1.0
        _p1[1, 1] = 1.0
        _p1[2, 2] = 1.0
        P1 = np.matmul(self.paramaters.K1, _p1)
        P2 = np.matmul(self.paramaters.K2, np.concatenate(
                (self.paramaters.R, self.paramaters.t.reshape(3, 1)), axis=1))

        self.paramaters.P1 = np.float64(P1)
        self.paramaters.P2 = np.float64(P2)

    def _load_params(self, paramLoader):
        """
        Set paramters to paramLoader
        """
        if paramLoader.K1[0]:
            self.paramaters.K1 = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.K1[1], delimiter=','))
        if paramLoader.K2[0]:
            self.paramaters.K2 = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.K2[1], delimiter=','))
        if paramLoader.d1[0]:
            self.paramaters.d1 = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.d1[1], delimiter=','))
        if paramLoader.d2[0]:
            self.paramaters.d2 = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.d2[1], delimiter=','))
        if paramLoader.R[0]:
            self.paramaters.R = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.R[1], delimiter=','))
        if paramLoader.t[0]:
            self.paramaters.t = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.t[1], delimiter=','))
        if paramLoader.P1[0]:
            self.paramaters.P1 = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.P1[1], delimiter=','))
        if paramLoader.P2[0]:
            self.paramaters.P2 = np.float64(np.loadtxt(
                paramLoader.base_path + paramLoader.P2[1], delimiter=','))
