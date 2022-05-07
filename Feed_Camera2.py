import time
import cv2
from util_functions import UtilFunctions

prevframe = 0
currframe = 0
# '''
poseDetector = UtilFunctions("Front")
# vidname = "Untitled (3).MOV"
# vidname = "VaibhavF2.mp4"
vidname = 1
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture(vidname)
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
frame_size = (frame_width,frame_height)
fps = 30

# Uncomment below to write to output file and use all outs below. Adjust
# out = cv2.VideoWriter(str(vidname)+"_output.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
while capture.isOpened():
    flag, frame = capture.read()
    if flag:
        currframe = time.time()
        fps = 1/(currframe-prevframe)
        fps = str(int(fps))
        prevframe = currframe
        img, landmarks = poseDetector.findPose(frame)
        #print(img.shape)
        barbell_color = poseDetector.slopeCheckBarbell(img, landmarks)
        lmList = poseDetector.findPosition(img, landmarks)
        # print(landmarks.landmark[27])
        result, barbell_color = poseDetector.generate_barbell_feedback(barbell_color)
        # poseDetector.draw_landmarks(img, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        poseDetector.display_feedback(img, result, "", barbell_color, barbell_color)
        # out.write(img)
        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(4) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
# out.release()

'''
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        currframe = time.time()
        fps = 1/(currframe-prevframe)
        fps = str(int(fps))
        prevframe = currframe
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        # putting the FPS count on the frame
        cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
'''