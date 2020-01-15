import cv2
import json
import glob

json_files = sorted(glob.glob('size_speed_test/*.json'))
for jfile in json_files:
    with open(jfile) as f:
        data = json.load(f)

        for key in data.keys():
            detection_data = data[key]
            fname1 = detection_data['fname1']
            fname2 = detection_data['fname2']
            img1 = cv2.imread(fname1.rstrip())
            img2 = cv2.imread(fname2.rstrip())
            cv2.imshow("img1", img1)
            cv2.waitKey(0)
            sq1_pose = detection_data['sq1_pose']
            sq2_pose = detection_data['sq2_pose']
            detection_size = detection_data['detection_size']
            detection_loc = detection_data['detection_loc']
            if 'motion' in detection_data.keys():
                motion = detection_data['motion']
