import cv2
import argparse


def main(args):
    images = args.images_file
    base_save_path = args.save_path
    file = open(images, 'r')
    count = 0
    for fnames in file:
        optical_img = cv2.imread(fnames.split(',')[0])
        sonar_img = cv2.imread(fnames.split(',')[1].rstrip())



        name_prefix = '_'.join(fnames.split(',')[1].rstrip().split('/')[4].split(' '))
        optical_save_path = base_save_path + "optical"
        optical_fname = "/optical" + name_prefix + "_" + str(count) + ".png"
        sonar_save_path = base_save_path + "acoustic"
        sonar_fname = "/acoustic" + name_prefix + "_" + str(count) + ".png"

        # f = open('optical_sonar_filenames.txt', 'a+')
        # line = optical_fname + ',' + sonar_fname + ',' + fnames.split(',')[0] + ',' + fnames.split(',')[1].rstrip() + '\n'
        # f.write(line)
        # f.close()

        #print(optical_save_path + optical_fname, sonar_save_path + sonar_fname)
        #cv2.imwrite(optical_save_path + optical_fname, optical_img)
        #cv2.imwrite(sonar_save_path + sonar_fname, sonar_img)

        count += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creation of co-registered \
        videos from AMP optical/acoustic data")
    parser.add_argument("--images_file", help="Path to images text file",
                        default="data/co-registered_detections.txt")
    parser.add_argument('--save_path',
                        help="path to save acoustic optical images",
                        default="/home/mitchell/WAMP_WS2/WAMP_co-registered_images/")

    args = parser.parse_args()

    main(args)
