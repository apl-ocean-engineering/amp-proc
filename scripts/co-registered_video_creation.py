import cv2
import argparse
import glob
from ampProc.amp_common import ampCommon
import numpy as np

def main(optical_path, acoustic_path, optical_filetype, acoustic_filetype):
    optical_images = sorted(glob.glob(optical_path + '*' + optical_filetype))
    acoustic_images = sorted(glob.glob(acoustic_path + '*' + acoustic_filetype))
    print(float(len(optical_images)), float(len(acoustic_images)))
    ac = ampCommon()

    optical_count = 0
    acoustic_count = 0

    img = cv2.imread(acoustic_images[0])

    height , width , layers =  img.shape

    video = cv2.VideoWriter('co-registered.avi',cv2.VideoWriter_fourcc(*"XVID"), 5, (width*2,height))
    #for acoustic_img in acoustic_images:
    while optical_count < len(optical_images) and acoustic_count < len(acoustic_images):
        optical_fname= optical_images[optical_count]
        acoustic_fname = acoustic_images[acoustic_count]
        if optical_count==0 and acoustic_count==0:
            previous_name_list = [optical_fname, acoustic_fname]
        print(optical_fname, acoustic_fname)
        time_diff = ac.relative_time_diff(optical_fname, acoustic_fname)
        print(time_diff, optical_count, acoustic_count)
        if time_diff < 0:
            optical_count+=1
        else:
            acoustic_count+=1
        if abs(time_diff) > 0.1:
            continue
        #print(optical_fname.split('/')[-1], acoustic_fname.split('/')[-1])
        name_list = [optical_fname, acoustic_fname]
        if optical_count>0 and acoustic_count>0:
            if name_list[0] == previous_name_list[0] or name_list[1] == previous_name_list[1]:
                #previous_name_list = name_list
                continue
        #print(name_list[0].split('/')[-1], name_list[1].split('/')[-1])
        previous_name_list = name_list
        optical_img = cv2.imread(optical_fname)
        acoustic_img = cv2.imread(acoustic_fname)
        optical_img = cv2.resize(optical_img,(int(acoustic_img.shape[1]),int(acoustic_img.shape[0])))
        print(optical_img.shape, acoustic_img.shape)
        vis = np.concatenate((optical_img, acoustic_img), axis=1)
        video.write(vis)
        cv2.imshow("vis", vis)
        #cv2.imshow("acoustic_img", acoustic_img)
        cv2.waitKey(100)

    video.release()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creation of co-registered \
        videos from AMP optical/acoustic data")
    parser.add_argument("optical_images", help="Path to optical images")
    parser.add_argument("acoustic_images", help="Path to acoustic images")
    parser.add_argument("--optical_filetype", help="Optical image encoding",
        default=".jpg")
    parser.add_argument("--acoustic_filetype", help="Acoustic image encoding",
        default=".jpg")

    args = parser.parse_args()

    optical_path = args.optical_images
    acoustic_path = args.acoustic_images
    if optical_path[-1] != '/':
        optical_path += '/'
    if acoustic_path[-1] != '/':
        acoustic_path += '/'


    main(optical_path, acoustic_path,
        args.optical_filetype, args.acoustic_filetype)
