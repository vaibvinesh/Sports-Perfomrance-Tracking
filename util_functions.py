import cv2
import mediapipe as mp
import math
from typing import List, Mapping, Optional, Tuple, Union
import numpy as np

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec, _normalized_to_pixel_coordinates

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (0, 255, 255)
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3


class UtilFunctions:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, view, mode=False, smooth=True,
                 detectionCon=0.6, trackCon=0.6):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param upBody: Upper boy only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """
        self.min_angle = 180
        self.max_distance = -1
        self.view = view
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.visibility = [-1 for i in range(34)]
        self.stable_values = [[0, 0, 0] for i in range(34)]

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon, model_complexity=1)

    def findPose(self, img, world=False):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if world:
            return np.array(img), results.pose_world_landmarks
        else:
            return np.array(img), results.pose_landmarks

    def draw(self, img, landmarks_list):
        if landmarks_list:
            self.mpDraw.draw_landmarks(img, landmarks_list,
                                       self.mpPose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                                       )

    def findPosition(self, img, landmarks):
        lmList = []
        if landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(landmarks.landmark):
                lmList.append([id, int(lm.x * w), int(lm.y * h), lm.visibility])
        return lmList

    def check_visibility(self, landmarks_list, check):
        if landmarks_list:
            return landmarks_list.landmark[check].visibility

    def update_landmark(self, landmarks_list, joint):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param landmarks_list: A normalized landmark list which needs to be updated
        :param joint: Flag to draw the output on the image.
        :ret : updated Landmarks_list
        """
        # This function updates the more stable value to the join under consideration
        if landmarks_list:
            if self.check_visibility(landmarks_list, joint) >= (
                    self.visibility[joint] - self.visibility[joint] * 0.008):
                self.stable_values[joint][0] = landmarks_list.landmark[joint].x
                self.stable_values[joint][1] = landmarks_list.landmark[joint].y
                self.stable_values[joint][2] = landmarks_list.landmark[joint].z

                self.visibility[joint] = self.check_visibility(landmarks_list, joint)

        landmarks_list.landmark[joint].x = self.stable_values[joint][0]
        landmarks_list.landmark[joint].y = self.stable_values[joint][1]
        landmarks_list.landmark[joint].z = self.stable_values[joint][2]

        return landmarks_list

    def drawLine(self, img, pt1, pt2, ptcolor, line_color, draw_points=True, r=8, t=2):
        cv2.line(img, pt1, pt2, line_color, t)
        if draw_points:
            cv2.circle(img, pt1, r, ptcolor, cv2.FILLED)
            cv2.circle(img, pt2, r, ptcolor, cv2.FILLED)

    def findAngle(self, img, p1, p2, p3, lmlist, draw=True):
        """
        Finds angle between three points. Inputs index values of landmarks
        instead of the actual points.
        :param img: Image to draw output on.
        :param p1: Point1 - Index of Landmark 1.
        :param p2: Point2 - Index of Landmark 2.
        :param p3: Point3 - Index of Landmark 3.
        :param draw:  Flag to draw the output on the image.
        :return:
        """

        # Get the landmarks
        x1, y1 = lmlist[p1][1:3]
        x2, y2 = lmlist[p2][1:3]
        x3, y3 = lmlist[p3][1:3]

        # Calculate the Angle
        angle = math.degrees(math.atan2((y3 - y2), (x3 - x2)) -
                             math.atan2((y1 - y2), (x1 - x2)))
        if angle < 0:
            angle+=360
        # Draw
        if draw:
            self.drawLine(img, (x1, y1), (x2, y2), RED_COLOR, WHITE_COLOR)
            self.drawLine(img, (x3, y3), (x2, y2), RED_COLOR, RED_COLOR)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findDistance(self, p1, p2, img, lmlist, draw=True, r=15, t=4):
        x1, y1 = lmlist[p1][1:3]
        x2, y2 = lmlist[p2][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            self.drawLine(img, (x1, y1), (x2, y2), (255, 0, 255), (255, 0, 255))

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def checkCurvature(self, p1, p2, img, lmlist, draw=True):
        x1, y1 = lmlist[p1][1:3]
        x2, y2 = lmlist[p2][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)
        self.max_distance = max(length, self.max_distance)
        #print(self.max_distance, length)
        if length + 25 >= self.max_distance:
            if draw:
                self.drawLine(img, (x1, y1), (cx, cy), GREEN_COLOR, GREEN_COLOR)
                self.drawLine(img, (x2, y2), (cx, cy), GREEN_COLOR, GREEN_COLOR)
            return GREEN_COLOR
        else:
            if draw:
                self.drawLine(img, (x1, y1), (cx, cy), RED_COLOR, RED_COLOR)
                self.drawLine(img, (x2, y2), (cx, cy), RED_COLOR, RED_COLOR)
            return RED_COLOR

    def angleCheck(self, img, p1, p2,p3, target, isleftView, lmlist, addOn=13, cutoff=45, draw=True):
        x1, y1 = lmlist[p1][1:3]
        x2, y2 = lmlist[p2][1:3]
        x3, y3 = x1, y2
        if isleftView:
            return self.angleCheckLeft(img, p1,p2,p3, target, lmlist,addOn, cutoff, draw)
        # Calculate the Angle
        angle = math.degrees(math.atan2((y3 - y2), (x3 - x2)) -
                             math.atan2((y1 - y2), (x1 - x2)))
        #print("here", angle)
        if -3 < angle < target + addOn and angle < cutoff:
            if draw:
                self.drawLine(img, (x3, y3), (x2, y2), GREEN_COLOR, GREEN_COLOR)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return GREEN_COLOR, angle
        elif angle < -3:
            if draw:
                self.drawLine(img, (x3, y3), (x2, y2), YELLOW_COLOR, YELLOW_COLOR)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return YELLOW_COLOR, angle
        elif angle < cutoff:
            if draw:
                self.drawLine(img, (x3, y3), (x2, y2), RED_COLOR, RED_COLOR)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            return RED_COLOR, angle
        else:
            return BLACK_COLOR, angle

    def generate_angle_Feedback(self, back_angle_color):
        result = []
        if back_angle_color == RED_COLOR:
            result.append("Don't raise your back too much")
        elif back_angle_color == YELLOW_COLOR:
            result.append("Keep your torso parallel to the floor")
        else:
            result.append("Good Form! keep you back in this range.")

        return result[0], back_angle_color

    def generate_back_feedback(self, back_color):
        result = []
        if back_color == RED_COLOR:
            result.append("Don't round/curve your spine!")
        else:
            result.append("Good form! spine is natural and not rounded")
        return result[0], back_color

    def display_feedback(self, img, result_back, result_back_angle, color_back, color_angle):
        cv2.rectangle(img, (0, 0), (560, 80), (0, 0, 0), -1)
        cv2.putText(img, result_back, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_back, 2, cv2.LINE_AA)
        cv2.putText(img, result_back_angle, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_angle, 2, cv2.LINE_AA)

    def angleCheckLeft(self, img, p1, p2, p3, target, lmlist, addOn, cutoff, draw):
        # Calculate the Angle
        angle = self.findAngle(img,p1,p2,p3,lmlist, draw = False)

        x1, y1 = lmlist[p1][1:3]
        x2, y2 = lmlist[p2][1:3]
        x3, y3 = lmlist[p3][1:3]

        if angle > self.min_angle == 180:
            self.min_angle = angle
        print(self.min_angle-angle)
        if -3 < self.min_angle-angle < target + addOn and self.min_angle-angle < cutoff:
            if draw:
                self.drawLine(img, (x3, y3), (x2, y2), GREEN_COLOR, GREEN_COLOR)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return GREEN_COLOR, self.min_angle-angle
        elif self.min_angle - angle < -3:
            if draw:
                self.drawLine(img, (x3, y3), (x2, y2), YELLOW_COLOR, YELLOW_COLOR)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return YELLOW_COLOR, self.min_angle-angle
        elif self.min_angle - angle < cutoff:
            if draw:
                self.drawLine(img, (x3, y3), (x2, y2), RED_COLOR, RED_COLOR)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            return RED_COLOR, self.min_angle-angle
        else:
            self.drawLine(img, (x3, y3), (x2, y2), (255,0,0), (255,0,0))
            return BLACK_COLOR, self.min_angle-angle

    def slopeCheckBarbell(self, img, landmarks, draw = True):
        img_h, img_w, _ = img.shape
        left_wrist = [int(landmarks.landmark[15].x*img_w), int(landmarks.landmark[15].y*img_h), int(landmarks.landmark[15].z*img_w)]
        right_wrist = [int(landmarks.landmark[16].x*img_w), int(landmarks.landmark[16].y*img_h), int(landmarks.landmark[16].z*img_w)]

        wrist_slope = pow((left_wrist[2] - right_wrist[2]), 2) / math.sqrt(
            pow((left_wrist[0] - right_wrist[0]), 2) + pow((left_wrist[1] - right_wrist[1]), 2))
        left_ankle = [landmarks.landmark[27].x, landmarks.landmark[27].y, landmarks.landmark[27].z]
        right_ankle = [landmarks.landmark[28].x, landmarks.landmark[28].y, landmarks.landmark[28].z]

        ankle_slope = pow((left_ankle[2] - right_ankle[2]), 2) / math.sqrt(
            pow((left_ankle[0] - right_ankle[0]), 2) + pow((left_ankle[1] - right_ankle[1]), 2))
        # print("slope of wrists:",wrist_slope)
        # print(wrist_slope - ankle_slope)
        if abs(wrist_slope - ankle_slope) < 2:
            if draw:
                self.drawLine(img, left_wrist[0:2], right_wrist[0:2], GREEN_COLOR, GREEN_COLOR, draw_points=False)
            return GREEN_COLOR
        else:
            if draw:
                self.drawLine(img, left_wrist[0:2], right_wrist[0:2], RED_COLOR, RED_COLOR, draw_points=False)
            return RED_COLOR

    def generate_barbell_feedback(self, barbell_color):
        result = []
        if barbell_color == RED_COLOR:
            result.append("The barbell is tilting keep it straight!")
        else:
            result.append("Good! Keep the barbell straight. ")
        return result[0], barbell_color
