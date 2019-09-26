# Calibration
Stores calibration paramaters here, which can be easily parsed into source code for mono/stereo image processing.   

See cfg/configCalibration.yaml for an examle yaml load file  

## Info
Each file is a .csv file which relates to camera intrisnics/extrinsics. Intrisics are locaded in camera1/ camera2/ and extrinsics are located in base folder  

## Intrinsics 
distortion_coeffs: 1X5 distortion coefficients matrix  
intrinsic_matrix.csv: 3X intrinsic K matrix   

## Extrinsics
E: 3X3 essential matrix  
F: 3X3 fundamental matrix  
P1: 3X4 projection matrix for camera 1  
P2: 3X4 projection matrix for camera 2  
Q: 4X4 disparity-to-depth mapping matrix  
R: 3X3 extrinsic rotation matrix  
R1: 3X3 rectification matrix for camera 1   
R2: 3X3 rectification matrix for camera 2   
t: 3X1 extrinsic translation vector   

