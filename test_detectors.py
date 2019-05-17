import cv2
import numpy as np 
from Detectors.YoloOpencvDetector import YoloOpencvDetetor
from Detectors import Utils 

# detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3.cfg", "./Detectors/YOLO/yolov3_320.weights")
# detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3_tiny.cfg", "./Detectors/YOLO/yolov3_tiny.weights")
detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3.cfg", "./Detectors/YOLO/yolov3_416.weights")
cap = cv2.VideoCapture("/home/vasily/Downloads/DJI_0002.MP4")
# out = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_3.avi',fourcc, 20.0, (1920,1080))
frame_i = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    if frame_i % 4 == 0:
        
        # frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        # boxes, classIDs, confidences = detector.detect(frame, s=(320, 320))
        # boxes, classIDs, confidences = detector.detect(frame, s=(416, 416))
        boxes, classIDs, confidences = detector.detect(frame, s=(700, 700))
        frame = Utils.draw_boxes(frame, boxes, classIDs, confidences, detector.CLASSES, COLORS=detector.COLORS)
        out.write(frame)
        cv2.imshow("frame", frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
    frame_i +=1 
cap.release()
out.release()