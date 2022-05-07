import time

import cv2
from util_functions import UtilFunctions
import mediapipe as mp
import numpy as np

RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (0, 255, 255)
ANGLE_CUTOFF = 50
FPS =30
IDX_LSHLDER = 11
IDX_RSHLDER = 12
IDX_LHIP = 23
IDX_RHIP = 24
IDX_LKNEE  = 25
IDX_RKNEE = 26

poseDetector = UtilFunctions("Side")
vidname = 0 # Put your camera ID here or video file name here
capture = cv2.VideoCapture(vidname)
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
frame_size = (frame_width, frame_height)

direction = 0
# Initialize below to write an output video to file
# out = cv2.VideoWriter(str(vidname)+"_output.mp4", cv2.VideoWriter_fourcc(*'XVID'), FPS, frame_size)
rep_start = False
result_back = ""
result_back_angle = ""
color_back = (0, 0, 255)
color_angle = (0, 0, 255)
min_angle = 180
prev_angle = 0
points = {}
viewTested = False


while capture.isOpened():
    flag, frame = capture.read()
    if flag:
        img, landmarks = poseDetector.findPose(frame)
        if not viewTested:
            isleftview = poseDetector.check_visibility(landmarks, IDX_LSHLDER) > poseDetector.check_visibility(landmarks, IDX_RSHLDER)
        viewTested = True
        # print(isleftview)
        if isleftview:
            points['knee'] = IDX_LKNEE
            points['shoulder'] = IDX_LSHLDER
            points['hip'] = IDX_LHIP
        else:
            points['knee'] = IDX_RKNEE
            points['shoulder'] = IDX_RSHLDER
            points['hip'] = IDX_RHIP
        landmarks = poseDetector.update_landmark(landmarks, points['knee'])
        lmList = poseDetector.findPosition(img, landmarks)

        back_color = poseDetector.checkCurvature(points['shoulder'], points['hip'], img, lmList, draw=True)
        back_angle_color, angle = poseDetector.angleCheck(img, points['shoulder'], points['hip'], points['knee'], 15, isleftview,
                                                          lmList, draw=True)
        min_angle = min(min_angle, angle)
        # print(min_angle, angle,  angle<40)
        if angle < 40:  # cut off for bent over person
            curr_posn = np.interp(angle, (min_angle, 20), (0, 100))
            if curr_posn == 100:
                # print(isleftview,curr_posn, direction, int(angle), int(prev_angle))
                if direction == 0:
                    direction = 1 if prev_angle > angle else 0
                    result_back_angle, color_angle = poseDetector.generate_angle_Feedback(back_angle_color)
            if curr_posn <= 20:
                # print(curr_posn,direction, int(angle), int(prev_angle))
                if direction == 1:
                    direction = 0 if prev_angle < angle else 1
                    result_back, color_back = poseDetector.generate_back_feedback(back_color)
                    back_angle, color = poseDetector.generate_angle_Feedback(back_angle_color)
                    if color == YELLOW_COLOR:
                        result_back_angle = back_angle
                        color_angle = color
        elif angle > 40:
            direction = 0
            result_back, result_back_angle = "", ""
        prev_angle = angle
        poseDetector.display_feedback(img, result_back, result_back_angle, color_back, color_angle)

        if not isleftview:
            poseDetector.drawLine(img, lmList[points['hip']][1:3], lmList[points['knee']][1:3], (255, 0, 0), (255, 0, 0))

        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(4) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
# out.release()
