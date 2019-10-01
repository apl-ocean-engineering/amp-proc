#!/usr/bin/env python3
from ampProc.amp_common import ampCommon
import cv2
import glob
import yaml
from os.path import dirname, abspath
from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
import argparse
import rospy
from sensor_msgs.msg import CameraInfo, Image
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

img1_pub = rospy.Publisher("manta/left/image_raw", Image, queue_size=10)
img2_pub = rospy.Publisher("manta/right/image_raw", Image, queue_size=10)

info1_pub = rospy.Publisher("manta/left/camera_info", CameraInfo, queue_size=10)
info2_pub = rospy.Publisher("manta/right/camera_info", CameraInfo, queue_size=10)

bridge = CvBridge()


def get_header(camera):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()

    if camera == 1:
        header.frame_id = "manta1"
    elif camera == 2:
        header.frame_id = "manta2"

    return header


def construct_info_message(im_size, K, D, R, P):
    camera_info_msg = CameraInfo()
    camera_info_msg.width = im_size[0]
    camera_info_msg.height = im_size[1]
    #print(K)
    camera_info_msg.K = list(K.reshape((1,9)))[0]
    camera_info_msg.D = list(D.reshape((1,5)))[0]
    camera_info_msg.R = list(R.reshape((1,9)))[0]
    camera_info_msg.P = list(P.reshape((1,12)))[0]
    camera_info_msg.distortion_model = "plumb_bob"

    return camera_info_msg


def main(args):
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
    # # Get rectification maps
    if EI_loader.paramaters.R1 is None or EI_loader.paramaters.R1 is None:
        EI_loader.calculate_rectification_matracies()

    new_k2, roi = cv2.getOptimalNewCameraMatrix(
        EI_loader.paramaters.K2, EI_loader.paramaters.d2, im_size, 1, im_size)

    map1, map2 = cv2.initUndistortRectifyMap(
            EI_loader.paramaters.K2, EI_loader.paramaters.d2,
            EI_loader.paramaters.R2, EI_loader.paramaters.P2[0:3, 0:3],
            im_size, cv2.CV_32FC1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        i = 0
        while i < len(images) and not rospy.is_shutdown():
            frame1, frame2 = ac.find_date(images, i)
            if frame1 is not None and frame2 is not None:
                img1 = cv2.imread(frame1)
                img2 = cv2.imread(frame2)

                # Publish messages
                cam_msg1 = construct_info_message(
                    im_size, EI_loader.paramaters.K1, EI_loader.paramaters.d1,
                    EI_loader.paramaters.R1, EI_loader.paramaters.P1)
                cam_msg2 = construct_info_message(
                    im_size, EI_loader.paramaters.K2, EI_loader.paramaters.d2,
                    EI_loader.paramaters.R2, EI_loader.paramaters.P2)

                header1 = get_header(1)
                header2 = get_header(2)

                cam_msg1.header = header1
                cam_msg2.header = header2

                info1_pub.publish(cam_msg1)
                info2_pub.publish(cam_msg2)

                rate.sleep()

                try:
                    ros_img1 = bridge.cv2_to_imgmsg(img1, "bgr8")
                    ros_img2 = bridge.cv2_to_imgmsg(img2, "bgr8")
                    ros_img1.header = header1
                    ros_img2.header = header2
                    img1_pub.publish(ros_img1)
                    img2_pub.publish(ros_img2)
                except CvBridgeError as e:
                    print(e)
            i += 1




if __name__ == '__main__':
    rospy.init_node("ros_manta_stream")
    parser = argparse.ArgumentParser("stream images and info over ROS")
    parser.add_argument("images", help="Base directory to images")
    parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(
            __file__))) + "/cfg/calibrationConfig.yaml")
    args = parser.parse_args()  # parse the command line arguments

    main(args)
