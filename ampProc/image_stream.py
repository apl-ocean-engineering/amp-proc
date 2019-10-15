#!/usr/bin/env python3
import cv2
import os
import time
from ampProc.constans import Constants
import copy

# class ImageStream:
#
#     def __init__(self, write_file_name, wait_time = 50, interesting_event_name = ' '):
#         self.wait_time = 50
#         self.time_inital = time.time()
#
#     def stream_images(self, img1, img_num, img2 = None):
#         # if img1 is not None:
#         #     cv2.imshow(Constants.img1_name, img1)
#         # if img2 is not None:
#         #     cv2.imshow(Constants.img2_name, img2)
#
#         # cv2.imshow('img', img)
#         k = cv2.waitKey(self.wait_time)
#         print("Time elapsed:", time.time() - self.time_inital)
#         print("Current file and file_num:", img_num)
#
#         if k == 112:  # p
#             print("PAUSING")
#             self.wait_time = 0
#             # return img_num
#
#         if k == 99:  # c
#             print("CONTINUING")
#             self.wait_time = 50
#             # return img_num
#
#         if k == 110:  # n
#             print("NEXT IMAGE")
#             img_num += 1
#             # return img_num
#
#         elif k == 98:  # b
#             print('GOING BACKWARDS')
#             '''
#             GO BACK
#             '''
#             img_num -= 1
#             # return img_num
#
#         elif k == 113:  # q
#             print('EXITING')
#             ''' Exit '''
#             exit()
#
#         img_num += 1
#
#
#         return img_num, k



class ImageStream:
    def __init__(self, write_file_name, wait_time = 50, interesting_event_name = ' '):
        self.wait_time = 50
        self.time_inital = time.time()

    def stream_images(self, img1, img_num, img2 = None):
        if img1 is not None:
             cv2.imshow(Constants.img1_name, img1)
        if img2 is not None:
             cv2.imshow(Constants.img2_name, img2)

        # cv2.imshow('img', img)
        k = cv2.waitKey(self.wait_time)
        print("Time elapsed:", time.time() - self.time_inital)
        print("Current file and file_num:", img_num)

        if k == 112:  # p
            print("PAUSING")
            self.wait_time = 0
            # return img_num

        if k == 99:  # c
            print("CONTINUING")
            self.wait_time = 50
            # return img_num

        if k == 110:  # n
            print("NEXT IMAGE")
            img_num += 1
            # return img_num

        elif k == 98:  # b
            print('GOING BACKWARDS')
            '''
            GO BACK
            '''
            img_num -= 1
            # return img_num

        elif k == 113:  # q
            print('EXITING')
            ''' Exit '''
            exit()

        img_num += 1


        return img_num, k

    def remove_line(self, write_file_name, img_file):
        write_file = open(write_file_name, 'r')
        file_lines = write_file.readlines()
        write_file.close()
        write_file = open(write_file_name, 'w+')
        for line in file_lines:
            if line != img_file:
                write_file.write(line)
        write_file.close()

    def loop_through_images(self, img, label):
        with open(label.strip()) as _label:
            for _bbox in _label:
                bbox = _bbox.split(' ')
                img = copy.copy(img)
                self._draw_rectangles(img, bbox)
                cv2.imshow(Constants.img1_name, img)
                cv2.waitKey(0)


    def display_labels(self, img, label, img_num):
        img_draw = copy.deepcopy(img)
        with open(label.strip()) as _label:
            for _bbox in _label:
                bbox = _bbox.split(' ')
                self._draw_rectangles(img_draw, bbox)

        img_num, k = self.stream_images(img_draw, img_num)
        if k == 13:
            self.loop_through_images(img, label)

        return img_num


    def _draw_rectangles(self, img, bbox):
        img_width = img.shape[1]
        img_height = img.shape[0]

        x_min = float(bbox[1])
        y_min = float(bbox[2])
        box_width = float(bbox[3])
        box_height = float(bbox[4])

        x = int(float(x_min - box_width/2)*img_width)
        y = int(float(y_min - box_height/2)*img_height)
        left_top = (x, y)
        x2 = int(float(x_min + box_width/2)*img_width)
        y2 = int(float(y_min + box_height/2)*img_height)
        bottom_right = (x2, y2)

        cv2.rectangle(img, left_top, bottom_right, (255,0,0), 2)
