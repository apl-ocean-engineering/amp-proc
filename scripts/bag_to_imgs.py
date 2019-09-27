#!/usr/bin/env python

"""
Created on Fri Mar  1 10:45:56 2019

@author: mitchell
"""
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse


class image_republish:
    def __init__(self, save_path):
        self.image_sub = rospy.Subscriber("/image_raw",
                                        Image, self.image_callback, 0)
        self.image_sub = rospy.Subscriber("/image_raw",
                                        Image, self.image_callback,1 )

        self.img = Image()
        self.bridge = CvBridge()

        self.save_path = save_path
        self.img_num = 0

    def image_callback(self, msg, name):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
          print(e)

        self.img_num += 1
        img_num = format(self.img_num, "05")
        if name == 0:
            img_name = self.save_path + "/Manta1" + img_num + ".png"
        if name == 1:
            img_name = self.save_path + "/Manta2" + img_num + ".png"

        cv2.imwrite(img_name, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Subscribe to images and save to folder")
    parser.add_argument(
        "save_path", help="Folder to save images")
    rospy.init_node("bag_to_png")
    IR = image_republish()
    rospy.spin()
