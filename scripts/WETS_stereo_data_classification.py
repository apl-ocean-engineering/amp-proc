import cv2

if __name__ == '__main__':
    path = "/home/mitchell/WAMP_WS2/AMP-proc/WETS_stereo_data_review/last_visted.txt"
    start = False
    start_time = "/home/mitchell/WAMP_workspace/WETS_stereo_test_images/2018_12_15/2018_12_15 10_39_47/Manta 1/2018_12_15_10_39_22.37.jpg"

    f = open(path, 'r')
    for line in f:
        fname1 = line.split(',')[2].rstrip()
        if fname1 == start_time:
            start = True
        if start:
            fname2 = fname1.replace('Manta 1', 'Manta 2')

            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)
            print(fname1)
            cv2.imshow("img1", img1)
            if fname2 is not None:
                cv2.imshow("img2", img2)

            k = cv2.waitKey(0)
            if k == 13:
                true_classifications = open('data/true_classifications.txt', 'a+')
                true_classifications.write(fname1 + '\n')
                true_classifications.close()
            else:
                false_classifications = open('data/false_classifications.txt', 'a+')
                false_classifications.write(fname1 + '\n')
                false_classifications.close()
