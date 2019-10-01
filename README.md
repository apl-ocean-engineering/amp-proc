# amp-proc
Contains processing code for AMP image processing

## Requirments
Python3  
Opencv  
Pytorch  
[Apl-OE pytorchYolo](https://github.com/apl-ocean-engineering/pytorchYolo)  
[stereo-processing](https://github.com/apl-ocean-engineering/stereo-processing)

## Instalation
$ git clone https://github.com/apl-ocean-engineering/amp-proc  
$ cd <amp-proc>  
$ pip3 install -e .  
$ pip install -e . (for python 2 support)  

## Scripts

### amp_yolo_stream.py
Using pytorchYolo and a trained deep learning image classification network,
this script will loop through all images in a specified dataset (TODO: Ease
user customizability) and instance segment predicted locations of objects
in the scene for each monocular camera.   
This process does not consider a monocular detection to neccessarily indicate
a positive trigger. Rather, it attempts to identify when either 1) many
fish are present, or 2) a fish is present in both scenes. Therefore a
trigger is raised when 1 of the following conditions is met:

1. 6 or more fish are present in one camera image
2. 3 or more fish are present in both camera images
3. A positive trigger from a rectified image analysis algorithm
   - This algorithm involves rectifying both stereo images, and
    searching along the x axis to verify a detection with a similar size
    has been found in a similar location
