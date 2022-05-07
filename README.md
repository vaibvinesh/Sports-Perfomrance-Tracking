# Sports Performance Tracking (CSE-598 Perception in Robotics)
This project was worked on under the guidance of Rajesh Aouti (github linked at the bottom). 

## Running Instructions:
1. Clone repo
2. You may update script based on input feed (Camera Id/ Input Video File)
3. Run Combined_Run.py (Make sure all scripts are in the same directory.

## Introduction:
In this project, we investigate several camera calibrations and posture estimation approaches. We then use our selected models to perform camera calibration and 3D reconstruction and investigate its applicability in real-time Human Pose Estimation. We also implemented Human Pose Estimation for the Bent Over Row at 20 fps using MediaPipe’s Pose module and used the heuristics provided to us, to critique and provide constructive feedback to the user. The report will address our methodology in detail as well detail our findings and conclusions.

![Capture](https://user-images.githubusercontent.com/67961257/167235982-5a3f9644-9b12-4a38-89dd-d10665577435.PNG)

## Implementation:
Estimating human poses from video plays an important role in this application. After careful consideration of the many pose estimators available, we decided to use MediaPipe for our application. MediaPipe Pose has a high fidelity body pose tracking ML solution that uses BlazePose Research, which also supports the ML Kit Pose Detection API, to derive 33 3D landmarks and background segmentation masks throughout the body from RGB video images[2]. Most of today’s modern approaches rely primarily on powerful desktop environments for inference, but they require low cost and high performance, and usually have low computing power at low cost. MediaPipe delivers real-time performance on modern mobile phones, desktops / laptops, Python, and even the web. MediaPipe’s Pose module’s high efficiency and low resource requirements was a key factor in our decision to use this model. Fig shows the pipeline for pose estimation in our implementation. 

### Pose and Landmark Extraction
To do this we call MediaPipe’s pose model on each frame from the input stream. The model named as BlazePose utilizes a lightweight CNN, which is customized to run on lightweight devices on real time. It extracts 33 key-points and provides their coordinates (based on both real world coordinates and image coordinates). This pose can be visualized in real-time on top of the existing video to perform skeletal tracking. The model is also able to dynamically detect and track only the most prominent figure in the video to avoid midsections and take utilize lesser resources. 

![FrontView](https://user-images.githubusercontent.com/67961257/167235983-40936cfc-2f05-4f55-a2d7-bff31ca0c111.PNG)

### Pose Tracking and Correctness
We use the landmarks obtained from the pose module to perform our analysis on pose correctness. The first step for side view is to detect the projection. To do this we use the Visibility metric provided for each key-point from MediaPipe. Visibility is a value between 0 and 1 that is a measure of the degree to which MediaPipe thinks the key-point can be seen based on camera POV. This metric is used to detect whether side view is taken from the left or from the right. Based on the detection of POV, we use the relevant landmarks to apply heuristics based on the exercises we are tracking for pose correctness. We do this after discussing with our industrial partner and expert on exercise pose. We track the pose while measuring the relevant heuristics to constantly provide relevant feedback by comparing to the correct measures obtained from the industry expert. For our project we track and critique the Bent Over row. Based on the provided heuristics we track 3 heuristics across both views. The key here is to decide which heuristic can be tracked in which POV. For example, the Bent Over row requires the user to keep a neutral spine, and this is tracked easily from the side view. The straightness of the barbell is another important heuristic, which can be tracked only from the front view.

![SideView](https://user-images.githubusercontent.com/67961257/167235986-160a6f6c-3610-4044-823a-f6f28ef6defe.png)

## References
MediaPipe Github: https://github.com/google/mediapipe
Rajesh Aouti Github: https://github.com/rajeshaouti/AI_GYM_ASSISTANT
