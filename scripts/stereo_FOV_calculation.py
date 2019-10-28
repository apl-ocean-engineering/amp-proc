import cv2
from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
import argparse


def main(args):
    loader = Loader(base_path=args.base_path)
    loader.load_params_from_file(args.calibration_yaml)
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader, img1.shape[:2])








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triangulation of AMP image \
                                     points")
    parser.add_argument("pixel_size",
        help="x y pixel size in micro meters", type=int)
    parser.add_argument("resolution",
        help="x y resolution in pixels", type=int)
    parser.add_argument("--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/cfg/calibrationConfig.yaml")
    parser.add_argument("--base_path",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/calibration/")

    args = parser.parse_args()
    main(args)
