import cv2
import numpy as np

# Load YOLOv4 network with OpenCV
net = cv2.dnn.readNet('/edgar1/weights-mng-raffi/ocrweightsversion4/MN.weights', '/edgar1/weights-mng-raffi/ocrweightsversion4/MN.cfg')

# Load COCO names
with open('/edgar1/openvino-object-detection/httpOCRpy/lib/MN/MN.names', 'r') as f:
    classes = [line.strip() for line in f]

# Load image
detection_img = cv2.imread('/edgar1/openvino-object-detection/media/3.png')

# Get image height and width
height, width = detection_img.shape[:2]

# Create a 4D blob from the image (normalize, scale, swap channels)
blob = cv2.dnn.blobFromImage(detection_img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Get the names of the output layers
output_layers = net.getUnconnectedOutLayersNames()

# Run forward pass to get predictions
detections = net.forward(output_layers)

thr = 0.1
prob_index = 5
full_result = ""
predict_plate = []
confidence_plate =[]
dets = []
# Process the detections
for out in detections:
    for detection in out:
        
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]#
        print(confidence)
        if confidence > thr:
            
            #nameTag = classNames[class_id]
            nameTag = classes[class_id]
            full_result = full_result + nameTag
            # Scale the bounding box back to the original image size
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            xRightTop  = x + w
            yRightTop  = y + h
            # Draw bounding box and label on the image
             #cv2.rectangle(detection_img, (x, y), (x + w, y + h), color, 2)
            print(nameTag,confidence)
            dets.append((nameTag, confidence, (x, y, w, h)))
            
            cv2.rectangle(detection_img, (x, y), (xRightTop, yRightTop),(0, 255, 0))
            if classes[class_id] in classes:
                label = classes[class_id] + ": " + str(round(confidence, 2))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                yLeftBottom = max(y, labelSize[1])
                cv2.rectangle(detection_img, (x, yLeftBottom - labelSize[1]),
                                        (x + labelSize[0], yLeftBottom + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                cv2.putText(detection_img, label, (x, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(detection_img, classes[class_id], (x, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.namedWindow('YOLOv4 Object Detection',0)
                cv2.imshow('YOLOv4 Object Detection', detection_img)
                cv2.waitKey(0)
                #cv2.destroyAllWindows()
