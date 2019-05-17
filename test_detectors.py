import cv2
import numpy as np 
from Detectors.YoloOpencvDetector import YoloOpencvDetetor
from Detectors import Utils 

detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3.cfg", "./Detectors/YOLO/yolov3_320.weights")

cap = cv2.VideoCapture("")


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    boxes, classIDs, confidences = detector.detect(frame, s=(320, 320))
    frame = Utils.draw_boxes(frame, boxes, classIDs, confidences, detector.CLASSES)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()