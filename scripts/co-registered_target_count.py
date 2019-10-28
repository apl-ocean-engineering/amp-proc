from os.path import dirname, abspath

base_path = dirname(dirname(abspath(__file__)))

sonar_detections_path = base_path + "/data/targets.csv"
optical_detections_path = base_path + "/data/3G_stream_YOLO_optical_dections_true.txt"


sonar_detections_file = open(sonar_detections_path, 'r')
optical_detections_file = open(optical_detections_path, 'r')

sonar_detections = sonar_detections_file.readlines()[1:]
optical_detections = optical_detections_file.readlines()

#print(sonar_detections[:3])
sonar_detections_date = ['_'.join(x.rstrip().split(',')[0].split(' '))[:-4] for x in sonar_detections]
sonar_detections_classification = [x.rstrip().split(',')[1] for x in sonar_detections]
sonar_detections_dictionary = {}

optical_detections_date = [x.rstrip().split('/')[-1][:-7] for x in optical_detections]

print(len(optical_detections_date), optical_detections_date[:3])

total_count = 0
correct_count = 0
for i, val in enumerate(optical_detections_date):
    if val in sonar_detections_date:
        classification = sonar_detections_classification[sonar_detections_date.index(val)]
        print(classification)
        total_count += 1
        if classification != "bubbles":
            correct_count+=1

print(total_count, correct_count)
