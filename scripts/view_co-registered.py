#!/usr/bin/env python2
import cv2
import argparse
import glob
from ampProc.video_streamer import StreamImages
from ampProc.amp_common import ampCommon
from os.path import dirname, abspath
import copy

OPTICAL_DISPLAY_NAME = "optical"
SONAR_DISPLAY_NAME = "sonar"

cv2.namedWindow(OPTICAL_DISPLAY_NAME, cv2.WINDOW_NORMAL)
cv2.namedWindow(SONAR_DISPLAY_NAME, cv2.WINDOW_NORMAL)


class StreamCoRegistered(StreamImages):

    def __init__(self):
        StreamImages.__init__(self)

    def main(self, args):
        ac = ampCommon(time_delay_allowed=0.2)
        folders_viewed = []

        f = open(args.interesting_events_txt_path, 'r')
        AMP_path = args.AMP_data_path
        for line in f:
            imgs = line.split(',')
            folder = '/'.join(imgs[0].split('/')[3:5])
            full_path = AMP_path + folder
            #print(full_path)
            if full_path not in folders_viewed:
                folders_viewed.append(full_path)
                Manta1_path = full_path + "/Manta 1"
                blue_view_path = full_path + "/BlueView"
                manta_imgs = sorted(glob.glob(Manta1_path + "/*.jpg"))
                sonar_imgs = sorted(glob.glob(blue_view_path + "/*.png"))
                img_count = 0
                sonar_count = 0
                #print(folder)
                # all_folders = open('M3_view_folders.txt', "a+")
                # all_folders.write(folder.split('/')[1] + '\n')
                # all_folders.close()

                while img_count < len(manta_imgs) and sonar_count < len(sonar_imgs):
                    fnames = ac.find_img_sonar(manta_imgs, sonar_imgs,
                                copy.copy(img_count), copy.copy(sonar_count))

                    img_count_increase, sonar_count_increase = self.display_images(
                            fnames[0], fnames[1], args.co_registered_save_path)
                    img_count += img_count_increase
                    sonar_count += sonar_count_increase

        print(folders_viewed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("View co-registered events")
    parser.add_argument(
        "--interesting_events_txt_path",
        help="Path to save interesting detections",
        default=dirname(dirname(abspath(
            __file__))) + "/data/interesting_events.txt")
    parser.add_argument(
        "--AMP_data_path",
        help="Path to save saved AMP data",
        default="/media/WAMP/")
    parser.add_argument(
        "--co_registered_save_path",
        help="Path to save saved co-registered data",
        default="data/co-registered_detections.txt")
    args = parser.parse_args()

    SC = StreamCoRegistered()
    SC.main(args)
