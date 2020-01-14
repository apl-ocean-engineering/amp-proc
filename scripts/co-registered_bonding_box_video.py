import json
import os
import argparse
import cv2
import numpy as np

IM_SIZE = (722, 2660, 3)

def main(args):
    text_file = open(args.images_path + args.images_txtfile, 'r')
    current_date = " "

    count = 0
    video = cv2.VideoWriter('co-registered_bboxes.avi',
        cv2.VideoWriter_fourcc(*"XVID"), 5, (IM_SIZE[1], IM_SIZE[0]))
    for line in text_file:
        optical_fname = args.images_path + "optical" + line.split(',')[0]
        acoustic_fname = args.images_path + "acoustic" + line.split(',')[1]
        optical_image = cv2.imread(optical_fname)
        acoustic_image = cv2.imread(acoustic_fname)
        cd = line.split(',')[-1].split('/')[-3]

        if current_date == cd:
            if optical_image is not None and acoustic_image is not None:
                _file = acoustic_fname.replace(".png", ".json")
                if os.path.exists(_file):
                    with open(_file) as json_file:
                        data = json.load(json_file)
                        if len(data["shapes"]) > 0:
                            for i in range(len(data["shapes"])):
                                label = data["shapes"][i]["label"]
                                points = data["shapes"][i]["points"]
                                left = (int(points[0][0]), int(points[0][1]))
                                right = (int(points[1][0]), int(points[1][1]))
                                width = float(abs((right[0] - left[0])))
                                height = float(abs((right[1] - left[1])))


                                cv2.rectangle(acoustic_image, left, right, [255, 255, 255])
                optical_image = cv2.resize(optical_image,(int(acoustic_image.shape[1]),int(acoustic_image.shape[0])))
                vis = np.concatenate((optical_image, acoustic_image), axis=1)
                if vis.shape != IM_SIZE:
                     vis = cv2.resize(vis, (IM_SIZE[1], IM_SIZE[0]))
                #print(vis.shape, vis.dtype)
                #cv2.imshow("optical", vis)
                #cv2.waitKey(1)
                video.write(vis)
        else:
            print(current_date, cd)
            current_date = cd
            for i in range(10):
                vis = np.zeros((IM_SIZE[1], IM_SIZE[0], 3))
                if vis.shape != IM_SIZE:
                    vis = cv2.resize(vis, (IM_SIZE[1], IM_SIZE[0]))
                cv2.putText(vis, current_date, (450, 350), cv2.FONT_HERSHEY_SIMPLEX,
                   5, [255, 255, 255], 2, cv2.LINE_AA)
                #print(vis.shape, IM_SIZE)
                vis = np.uint8(vis)
                #cv2.imshow("optical", vis)
                #cv2.waitKey(1)
                #print(vis.shape, vis.dtype)
                video.write(vis)
                #print(vis.shape)
            #cv2.imshow("acoustic", acoustic_image)
    video.release()







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creation of co-registered \
        videos from AMP optical/acoustic with bbox data")

    parser.add_argument("--images_path",
        help="path to images/acoustic folders",
        default="/home/mitchell/WAMP_WS2/WAMP_co-registered_images/")
    parser.add_argument("--images_txtfile",
        help="textfile with co-reg image information",
        default="optical_sonar_filenames.txt")

    args = parser.parse_args()
    main(args)
