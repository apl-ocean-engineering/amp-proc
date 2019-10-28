#!/usr/bin/env python3
from ampProc.amp_common import ampCommon
import cv2
import glob
from os.path import dirname, abspath
import argparse
import rospy
from sensor_msgs.msg import Image
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

img1_pub = rospy.Publisher("manta/left/image_raw", Image, queue_size=10)
img2_pub = rospy.Publisher("manta/right/image_raw", Image, queue_size=10)

manta1_folder = '/Manta 1/'
manta2_folder = '/Manta 2/'

bridge = CvBridge()

def get_header(camera):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()

    if camera == 1:
        header.frame_id = "manta1"
    elif camera == 2:
        header.frame_id = "manta2"

    return header

def main(base_path):
    if base_path[-1] != '/':
        base_path += '/'
    sub_folders = sorted(glob.glob(base_path + '*'))
    for folder in sub_folders:
        print(folder)
        manta1 = sorted(glob.glob(folder + manta1_folder + '*.bmp'))
        manta2 = sorted(glob.glob(folder + manta2_folder + '*.bmp'))
        images = []
        #print(manta1, manta2)
        for fname1, fname2 in zip(manta1, manta2):
            images.append((fname1, fname2))
        ac = ampCommon()
        ac.display = False
        i = 0

        rate = rospy.Rate(5)
        while i < len(images) and not rospy.is_shutdown():
            frame1, frame2 = ac.find_date(images, i)

            if frame1 is not None and frame2 is not None:
                #print(frame1)
                img1 = cv2.imread(frame1)
                img2 = cv2.imread(frame2)

                #cv2.imshow('img1', img1)
                #cv2.waitKey(1)

                header1 = get_header(1)
                header2 = get_header(2)

                try:
                    ros_img1 = bridge.cv2_to_imgmsg(img1, "bgr8")
                    ros_img2 = bridge.cv2_to_imgmsg(img2, "bgr8")
                    ros_img1.header = header1
                    ros_img2.header = header2
                    img1_pub.publish(ros_img1)
                    img2_pub.publish(ros_img2)
                except CvBridgeError as e:
                    print(e)
                rate.sleep()

            i+= 1


if __name__ == '__main__':
    rospy.init_node("ros_manta_stream")
    parser = argparse.ArgumentParser("stream images and info over ROS")
    # parser.add_argument("base_path", help="Base directory to images")
    #args = parser.parse_args()  # parse the command line arguments

    base_path = "/home/mitchell/Dropbox (MREL)/Camera Calibration Data/"

    main(base_path)
