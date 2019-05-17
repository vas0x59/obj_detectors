import cv2
import numpy as np 
from Detectors.YoloOpencvDetector import YoloOpencvDetetor
from Detectors import Utils 

detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3.cfg", "./Detectors/YOLO/yolov3_320.weights")

cap = cv2.VideoCapture("/home/vasily/Downloads/DJI_0002.MP4")
frame_i = 0

while True:
    ret, frame = cap.read()
    if frame_i % 24 == 0:
        if ret == False:
            break
        boxes, classIDs, confidences = detector.detect(frame, s=(320, 320))
        frame = Utils.draw_boxes(frame, boxes, classIDs, confidences, detector.CLASSES)

        cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    frame_i +=1 
cap.release()