import cv2
import numpy as np
import cvzone
import math
from sort import *

## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


## Model Files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confThreshold = 0.5
nmsThreshold= 0.2 # lower: more aggressive

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

def detect_object(image,predictions,confThreshold,nmsThreshold):
    detections = np.empty((0, 5))
    H,W,C = image.shape
    xyxy=[]
    bbox = []
    classIds = []
    confs = []
    for output in predictions:  #output is list of eg: 300 rows and 85 columns
        for det in output:  # elements in one row: 85 elements
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if classNames[classId] == "bicycle" and confidence > confThreshold:

                width,height = int(det[2]*W) , int(det[3]*H) # det[2] is in %, to get pixel value multiply by original width
                x,y = int((det[0]*W)-width/2) , int((det[1]*H)-height/2) # x,y: center of bbox
                x2,y2 = width + x, height + y
                bbox.append([x,y,width,height])
                classIds.append(classId)
                confs.append(float(confidence))
                xyxy.append([x,y,x2,y2])

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        #print(indices)

    for i in indices:
        conf=confs[i]
        box = xyxy[i]
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        currentArray = np.array([x1, y1, x2, y2, conf])
        detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(image,(0,220),(640,220),(0,0,255),1)
    for result in resultsTracker: 
        x1, y1, x2, y2, id = result

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(image, (x1, y1, w, h),l=15,t=2,colorR=(0,0,0))

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if 0 < cx < 640 and 240 - 30 < cy < 240 + 30:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(image,(0,220),(640,220),(0,255,0),1)

        cvzone.putTextRect(image, f'Count: {len(totalCount)}', (40, 40), scale=1.7, thickness=2,colorT=(0, 255, 0), colorR=(0, 0, 0))
        cv2.imshow('Image',image)
        cv2.waitKey(1) 

video = cv2.VideoCapture('./cycle.mp4')
ret, frame = video.read()
H, W, _ = frame.shape
out = cv2.VideoWriter('./result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(video.get(cv2.CAP_PROP_FPS)), (W, H))

totalCount = []
while ret:

    inputs = cv2.dnn.blobFromImage(frame, 1 / 255.0, (604, 604), swapRB = True, crop = False)
    net.setInput(inputs)
    layersNames = net.getLayerNames()

    index=net.getUnconnectedOutLayers()
    outputNames = [layersNames[index[0]-1],layersNames[index[1]-1],layersNames[index[2]-1]]

    predictions = net.forward(outputNames)
    detect_object(frame,predictions,confThreshold,nmsThreshold)
    #cv2.imshow('frame',cv2.resize(image,(1200,900), interpolation = cv2.INTER_LINEAR))
    #cv2.waitKey(1)  
    out.write(frame)
    ret, frame = video.read()

video.release()
out.release()
cv2.destroyAllWindows()



