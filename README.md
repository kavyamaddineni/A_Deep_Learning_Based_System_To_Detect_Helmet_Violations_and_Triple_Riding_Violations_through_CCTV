# Triple Riding and Helmet Violation Detection System Using YOLOv8

## Overview
This project implements a **Deep Learning-based system** for detecting **Triple Riding** (three people riding the same vehicle) and **Helmet Violations** (riders not wearing helmets) in real-time, using the **YOLOv8** object detection model. The system processes video streams from CCTV cameras or webcams to identify and highlight these violations, helping to improve road safety.

## Features
- Detects **Triple Riding** incidents (when three people are riding a single vehicle).
- Detects **Helmet Violations** (when riders are not wearing helmets).
- **Real-time** detection and visualization of violations.
- Built using **YOLOv8**, a state-of-the-art deep learning model for object detection.
- Supports input from **CCTV cameras** or **webcams** for live monitoring.

## Installation

### Step 1: Clone the Repository
Clone the repository to your local machine using the following command:

```
git clone https://github.com/kavyamaddineni/A_Deep_Learning_Based_System_To_Detect_Triple_Riding_and_Helmet_Violations_Through_CCTV_Webcam.git
```
### Navigate to the project directory:

```
cd A_Deep_Learning_Based_System_To_Detect_Triple_Riding_and_Helmet_Violations_Through_CCTV_Webcam
```
### Step 2: Install Dependencies
Install the required dependencies listed in the requirements.txt file.  
Run the following command to install them:
```
pip install -r requirements.txt
```
The **requirements.txt** should contain the following libraries:

```
torch==1.12.1
opencv-python==4.5.5.64
yolov8==0.1.0
numpy==1.21.0
matplotlib==3.5.1
```
### Step 3: Download YOLOv8 Pretrained Weights
YOLOv8 requires pretrained weights for object detection.  

Download the pretrained weights by running:
```
python -m yolov8.download
```
This will automatically download the necessary weights for YOLOv8.

### Step 4: Set up Dataset (Optional)
If you wish to train the model with your own data, place the dataset in the /data folder. Ensure that the dataset is in YOLO format (images and corresponding annotations).

You can use a tool like LabelImg to label images for detection tasks.

### Step 5: Train the Model (Optional)
If you want to train the YOLOv8 model with your own dataset, run the following command:

```
python train.py --data data/triple_riding_and_helmet.yaml --cfg yolov8.yaml --weights yolov8.weights
```
This will start training the model using your custom dataset.

**data/triple_riding_and_helmet.yaml:** Path to your data configuration file.
yolov8.yaml: YOLOv8 configuration file.
yolov8.weights: Path to the pretrained weights.  

### Step 6: Run the Detection
Once the model is trained, or if you want to use the pretrained model, you can run detection on video streams or images. Use the following command to run detection on a webcam feed:

```
python detect.py --source 0 --weights runs/train/exp/weights/best.pt
```
 

The best.pt file refers to the trained model weights that you obtained after training.

### Step 7: View the Results
Once the detection is running, the system will visualize the output in real-time, showing bounding boxes around detected Triple Riding and Helmet Violations. These boxes will help highlight the violators in the video stream.

Triple Riding will be highlighted with a bounding box around the riders.
Helmet Violations will be indicated if any of the riders are not wearing helmets.
You can stop the detection at any time by pressing Ctrl+C.


In the output, bounding boxes are drawn around detected violations, and the system marks the number of riders and identifies helmet violations if present.



Acknowledgments
YOLOv8: https://github.com/ultralytics/yolov8
OpenCV: Used for real-time image processing and video analysis.
PyTorch: For training the deep learning model.
Special thanks to all contributors and open-source developers who made this project possible.
vbnet
Copy

---

### Summary of Key Points:
1. **Installation Instructions**: Clear steps for cloning the repository, installing dependencies, and setting up YOLOv8.
2. **Training Instructions**: Instructions on how to train the model with your custom dataset (optional).
3. **Running the Model**: Instructions on how to run the model on real-time video streams or pre-recorded footage.
4. **Model Evaluation**: Instructions on how to evaluate the model's performance after training.
7. **Acknowledgments**: Recognition of libraries or frameworks used in the project.

 
