import cv2

true_list = open('true_classifications.txt', 'r')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)

for detection in true_list:
    img = cv2.imread(detection.rstrip())
    img2 = cv2.imread(detection.rstrip().replace('Manta 1', 'Manta 2'))
    print(detection.rstrip())
    print(detection.rstrip().replace('Manta 1', 'Manta 2'))

    cv2.imshow('img', img)
    if img2 is not None:
        cv2.imshow('img2', img2)
    k = cv2.waitKey(0)
    if k == 13:
        only_stereo = open('stereo_true_classifications.txt', 'a+')
        only_stereo.write(detection.rstrip() +'\n')
        only_stereo.close()
