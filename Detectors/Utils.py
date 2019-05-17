import cv2





def draw_boxes(img, boxes, ids, confs, CLASSES):
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # x /= 0.35
        # y /= 0.35
        # w /= 0.35
        # h /= 0.35
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(image_full, (x, y), (x + w, y + h), (0, 255, 150), 2)
    return img
