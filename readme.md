# Vehicle Detecction and Tracking( Yolov8 + SORT )

In this repository I have trained yolov8m on KITTI Dataset and urban Traffic Surveillance (UTS) dataset for vehicle Detection and used SORT Tracking algorithm for tracking vehicles

## Tracking 

Custom object detection architecture

The trackers provided in this repo can be used with other object detectors than Yolov8.

For tracking please run below file

```bash
$ python Track.py
```


To obtain vehicle counting results on distinct runways, utilize the "Detection_tracking_inference.ipynb" file.

# Detection and Tracking results
[![Vehicle Detection and Counting](clipped_joined_video.gif)](https://youtu.be/AisUbJo4vYQ)

[![Vehicle Tracking](./runs/detect/train/tracking.png)](https://youtu.be/UvWd7B-HVKo)

## Introduction

# KITTI Dataset
KITTI is 2D object detection and orientation estimation benchmark Dataset; it consists of 7481 training images and 7518 testing images. The benchmark uses 2D bounding box overlap to compute precision-recall curves for detection and computes orientation similarity to evaluate the orientation estimates in bird's eye view.
Kitti datsets is captured by driving around the mid-size city of Karlsruhe, in rural areas and on highways. Up to 15 cars and 30 pedestrians are visible per image.

![image](./runs/kitti_2d.png)

# Urban Traffic Surveillance (UTS) Dataset

The Urban Traffic Surveillance (UTS) dataset is collected with six videos which were captured from different viewpoints and illumination conditions. All the vehicles are annotated with bounding box, viewpoint, type and color. 

# Training and Data Preparation

The included code, which is in form of Ipython Notebook, downloads the dataset and perform processing.

# Colab Notebook
The code needed to preprocess the data and train the model is available in the following notebook.
Set the runtime to GPU and run everything.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/muhammad-umair-usmani/Human_Detection/blob/main/vehicle_dataset_preparation+yolov8_training.ipynb)

# Yolo Weights and checkpoint to resume training

The [weights](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/muhammad-umair-usmani/Human_Detection/blob/main/runs/detect/train) are also available to run using the official yolov8 code found [here](https://github.com/ultralytics/ultralytics)

you can find trained model checkpoint, tensorboard logs and weights at path "./runs/detect/train/"

Finally yolov8m is trained. Below are training metrics and performance images

<p float="left">
  <img src="./runs/detect/train/confusion_matrix.png" width="400" />
  <img src="./runs/detect/train/confusion_matrix_normalized.png" width="400" />
</p>
<p float="left">
  <img src="./runs/detect/train/F1_curve.png" width="200" />
  <img src="./runs/detect/train/P_curve.png" width="200" />
  <img src="./runs/detect/train/PR_curve.png" width="200" />
  <img src="./runs/detect/train/R_curve.png" width="200" />
</p>


# labels correlogram
<p float="left">
  <img src="./runs/detect/train/labels_correlogram.jpg" width="400" />
  <img src="./runs/detect/train/labels.jpg" width="400" />
</p>

# Results
![image](./runs/detect/train/train_batch21782.jpg)
![image](./runs/detect/train/val_batch2_pred.jpg)