import cv2
from pytorchYolo.argLoader import ArgLoader
from os.path import dirname, abspath
import glob
from ampProc.amp_common import ampCommon
from ampProc.video_streamer import StreamImages
import copy
from pytorchYolo.detector import YoloLiveVideoStream

OPTICAL_DISPLAY_NAME = "optical"
SONAR_DISPLAY_NAME = "sonar"

cv2.namedWindow(OPTICAL_DISPLAY_NAME, cv2.WINDOW_NORMAL)
cv2.namedWindow(SONAR_DISPLAY_NAME, cv2.WINDOW_NORMAL)


def main(detections_path, detector):
    ac = ampCommon()
    SI = StreamImages()
    f = open(detections_path, 'r')
    sonar_list = f.readlines()
    prev_folders = []
    img_count = 0
    sonar_count = 0
    #for line in f:
    while sonar_count < len(sonar_list):
        line = sonar_list[sonar_count]
        sonar_fname = line.split(',')[-1].rstrip()
        sonar_img = cv2.imread(sonar_fname)
        detection_fname = line.split(',')[-1].rstrip()
        folder = '/'.join(line.split(',')[-1].split('/')[:-2])
        if detector is not None:
            detection, yolo_squares = detector.stream_img(
                            sonar_img)
            #cv2.imshow("sonar", sonar_img)
            #cv2.waitKey(100)
        #if folder not in prev_folders:
        #print(folder)
        #prev_folders.append(folder)
        M1_images = sorted(glob.glob(folder + "/Manta 1/*.jpg"))
        #print(sonar_fname)
        #BV_images = sorted(glob.glob(folder + "/BlueView/*.png"))
        #while img_count < len(M1_images):

        # sonar_fname, manta_folders = ac.find_all_images_near_sonar(
        #                            M1_images, BV_images,
        #                            copy.copy(sonar_count))
        # #print(manta_folders)
        # if len(manta_folders) > 0:
        #     for manta_fname in manta_folders:
        #
        #         img_count_increase, sonar_count_increase = SI.display_images(
        #                 sonar_fname, manta_fname, yolo_detector = detector,
        #                 detection_fname=detection_fname)
        #
        #         img_count += img_count_increase
        #         sonar_count += sonar_count_increase
        #
        # else:
        #     img_count_increase, sonar_count_increase = SI.display_images(
        #             None, sonar_fname, yolo_detector = detector,
        #             detection_fname=detection_fname)
        #
        #     img_count += img_count_increase
        #     sonar_count += sonar_count_increase
        # sonar_fname = BV_images[sonar_count]
        #print(sonar_fname)
        manta_fname = ac.find_image_near_sonar(sonar_fname, M1_images)
        img_count_increase, sonar_count_increase = SI.display_images(
                manta_fname, sonar_fname, yolo_detector = detector,
                detection_fname=detection_fname)
        #print(img_count_increase, sonar_count_increase)
        img_count += img_count_increase
        sonar_count += sonar_count_increase


if __name__ == '__main__':
    argLoader = ArgLoader()
    sname = "/WAMP_acoustic_review/detection.txt"
    argLoader.parser.add_argument("--detections_path",
                                  help="Detections textfile path",
                                  default=dirname(
                                        dirname(abspath(__file__))) + sname)
    argLoader.parser.add_argument("--run_acoustic_yolo",
                                  help="run YOLO network on acoustic images",
                                  default=False)

    args = argLoader.args
    detections = args.detections_path

    if args.run_acoustic_yolo:
        detector = YoloLiveVideoStream(args)
        detector.display = False
    else:
        detector = None

    main(detections, detector)
