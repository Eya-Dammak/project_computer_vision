#Fruit Detection and Ripeness Estimation System
A computer vision system that detects and classifies fruits in real-time and evaluates their ripeness using color analysis. This project leverages YOLOv8 for object detection and HSV-based color segmentation for ripeness estimation, making it suitable for applications in smart agriculture, supply chain automation, and quality control.



##Project Overview

This project is built to:

-Detect multiple types of fruits in images or video streams.
-Determine the ripeness stage of each detected fruit.
-Count and summarize detected fruits by type and ripeness.
-Support real-time video processing and optional output recording.
-Enhance detection robustness through image preprocessing and augmentation.


##Functionalities
Fruit Detection:Uses a YOLOv8 object detection model to detect apples, oranges.... 
Ripeness Estimation:Classifies each detected fruit as "Ripe", "Unripe", or "Intermediate" based on HSV color range analysis. 
Image Preprocessing:Applies Contrast Limited Adaptive Histogram Equalization to improve image contrast before detection. 
Video Processing:Processes video streams frame-by-frame with real-time detection and optional recording of annotated output. 
Data Summary:Generates per-frame fruit statistics including type, count, and ripeness distribution. 
Visual Feedback:Displays labeled bounding boxes showing fruit type, confidence, and ripeness state. 

##Technologies Used

-Python 
-Ultralytics YOLOv8
-OpenCV
-NumPy

##How to deploy:
install the requirements 
run the following command
    python app.py
