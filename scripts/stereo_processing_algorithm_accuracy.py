import cv2

true_classification_list = "/home/mitchell/WAMP_WS2/AMP-proc/WETS_stereo_data_review/3G_pool_calibration/true_classifications.txt"
detection_list = "/home/mitchell/WAMP_WS2/AMP-proc/WETS_stereo_data_review/3G_pool_calibration/detection.txt"

true_classifications = open(true_classification_list, 'r').readlines()
detections = open(detection_list, 'r').readlines()

true_classifications = [x.rstrip() for x in true_classifications]
detections = [x.rstrip() for x in detections]
#print(true_classifications[:3])
#print(detections[:3])

true_positive_count = 0
total_positives = len(detections)
total_detections = len(true_classifications)
for i, line in enumerate(detections):
    detection = detections[i].split(',')[-2]
    print(detection)
    #classification = true_classifications[i]
    if detection in true_classifications:
        true_positive_count+=1
    else:
        img = cv2.imread(detections[i].split(',')[-2].rstrip())
        cv2.imshow("img", img)
        img2 = cv2.imread(detections[i].split(',')[-2].rstrip().replace('Manta 1', 'Manta 2'))
        if img2 is not None:
            cv2.imshow("img2", img2)
        cv2.waitKey(0)
    #print()
    #print(detection)
    #print(classification)
    #print('/home/mitchell/WAMP_workspace/WETS_stereo_test_images/2018_10_17/2018_10_17 12_58_19/Manta 1/2018_10_17_12_58_10.52.jpg' in true_classifications)

precision = true_positive_count/float(total_positives)
recall = true_positive_count/float(total_detections)

print('precision')
print(precision)

print('recall')
print(recall)

"""
incorrect
"""
