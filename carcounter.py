import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort

# Video Capture
cap = cv2.VideoCapture("../Videos/cars.mp4")

# Load YOLO Model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class Names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load Mask (if exists)
mask_path = "mask.png"
mask = cv2.imread(mask_path) if cv2.imread(mask_path) is not None else None

# Tracker Initialization
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Line Limits for Counting
limits = [400, 297, 673, 297]
totalCount = set()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break  # Stop if video ends
    
    if mask is not None:
        imgRegion = cv2.bitwise_and(img, mask)
    else:
        imgRegion = img.copy()
    
    # Overlay Graphics
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
    # YOLO Detection
    results = model(imgRegion, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                detections.append([x1, y1, x2, y2, conf])
    
    # Convert to NumPy Array
    detections = np.array(detections)

    # Update Tracker
    resultsTracker = tracker.update(detections)

    # Draw Line for Counting
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Draw Bounding Box & ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'ID: {id}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=5)
        
        # Draw Center Point
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # Vehicle Counting
        if limits[0] < cx < limits[2] and (limits[1] - 15) < cy < (limits[1] + 15):
            if id not in totalCount:
                totalCount.add(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    
    # Display Count
    cv2.putText(img, f'Count: {len(totalCount)}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 6)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
