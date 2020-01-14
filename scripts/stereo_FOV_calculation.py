import cv2
from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
from os.path import dirname, abspath
import argparse
import numpy as np

def blind_spot_calc(translation, h_fov1, h_fov2):
    #FOV in rads
    blind_spot_height = translation*\
        np.sin(np.pi/2 - h_fov1)*np.sin(np.pi/2 - h_fov2)/np.sin(h_fov1 + h_fov2)

    print(blind_spot_height)


def trans_from_P(projMat, K=None):
    if K is not None:
        new_K, R, t, Rx, Ry, Rz, _ = cv2.decomposeProjectionMatrix(projMat, K)
    else:
        new_K, R, t, Rx, Ry, Rz, _ = cv2.decomposeProjectionMatrix(projMat)

    return K, R, t

def calculate_FOV(k, im_size, x_size, y_size):
    fovx, fovy, focal_length, principal_point, aspect_ratio = \
        cv2.calibrationMatrixValues(k, im_size, x_size, y_size)

    return fovx, fovy


def main(args):
    loader = Loader(base_path=args.base_path)
    loader.load_params_from_file(args.calibration_yaml)
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)
    x_size = int(args.pixel_size[0]*EI_loader.paramaters.im_size[0]*1000000)
    y_size = int(args.pixel_size[1]*EI_loader.paramaters.im_size[1]*1000000)
    k = EI_loader.paramaters.K1
    im_size = (
        int(EI_loader.paramaters.im_size[0]),
        int(EI_loader.paramaters.im_size[1]))

    print("Camera1 FOV: ")
    fovx1, fovy1 = calculate_FOV(k, im_size, x_size, y_size)
    print(fovx1, fovy1)
    new_k1, R1, t1 = trans_from_P(EI_loader.paramaters.P1, K=k)

    k = EI_loader.paramaters.K2
    print("Camera2 FOV: ")
    fovx2, fovy2 = calculate_FOV(k, im_size, x_size, y_size)
    print(fovx2, fovy2)
    new_k2, R2, t2 = trans_from_P(EI_loader.paramaters.P2, K=k)
    t2/=t2[3]
    print("Camer2 trans: ")
    print(R2, t2[:3])

    blind_spot_calc(t2[0], np.radians(fovx1), np.radians(fovx2))














if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triangulation of AMP image \
                                     points")
    parser.add_argument("--pixel_size",
        help="x y pixel size in micro meters", nargs='+', type=float)
    parser.add_argument("--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/cfg/calibrationConfig.yaml")
    parser.add_argument("--base_path",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/calibration/")


    args = parser.parse_args()
    main(args)
