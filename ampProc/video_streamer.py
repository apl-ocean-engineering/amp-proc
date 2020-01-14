import cv2


class StreamImages:
    def __init__(self):
        self.wait_time = 100

    def display_images(self, optical_fname, sonar_fname, save_name=' ', yolo_detector = None, detection_fname = ' '):
        optical_img = cv2.imread(optical_fname)
        sonar_img = cv2.imread(sonar_fname)

        if sonar_img is None:
            return 1, 1
        if yolo_detector is not None:
            detection, yolo_squares = yolo_detector.stream_img(
                            sonar_img)
        #
        #print(detection_fname)
        if optical_img is not None:
            cv2.imshow("optical", optical_img)
        if sonar_img is not None:
            cv2.imshow("sonar", sonar_img)
        if sonar_fname == detection_fname:
            k = cv2.waitKey(self.wait_time)
        else:
            k = cv2.waitKey(self.wait_time)

        if k == 112:  # p
            print("PAUSING")
            self.wait_time = 0

        if k == 99:  # c
            print("CONTINUING")
            self.wait_time = 50

        if k == 113:  # q
            exit()

        if k == 13:  # enter
            pass
            # if save_name != ' ':
            #     f = open(save_name, "a+")
            #     save_name = optical_fname + "," + sonar_fname + "\n"
            #     f.write(save_name)
            #     f.close()

        if k == 98:  # b
            return -1, -1

        return 1, 1
