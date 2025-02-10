Vehicle Tracking and Counting Using YOLOv8 and SORT
This project implements real-time vehicle detection, tracking, and counting using YOLOv8 and SORT (Simple Online and Realtime Tracker). It processes a video feed to identify and track vehicles crossing a predefined counting line.

Features 🚗🚛🚌🏍️
Detects and tracks cars, trucks, buses, and motorbikes.
Assigns a unique ID to each detected vehicle.
Counts vehicles that cross a defined line.
Displays bounding boxes, object IDs, and the total count.
Uses a mask to focus on a specific region (optional).
Requirements 📌
Python
OpenCV
YOLOv8 (Ultralytics)
SORT tracker
cvzone for easy visualization
NumPy for data processing
How It Works 🏃‍♂️
Reads a video feed and applies YOLOv8 for object detection.
Filters out non-vehicle objects and keeps only relevant ones.
Tracks each vehicle across frames using the SORT tracker.
Draws bounding boxes and IDs for tracked vehicles.
Counts vehicles that cross a defined line on the screen.
Displays the total count of unique vehicles detected.
Customization ⚙️
Change the tracking line position to fit different camera angles.
Adjust the confidence threshold for better accuracy.
Modify the object classes to track other vehicles or objects.
Use a custom mask image to define specific detection areas.
Limitations & Considerations 🚦
Works best in good lighting conditions.
Overlapping vehicles might cause tracking issues.
Camera angle affects detection and counting accuracy.
Can be improved with DeepSORT for more robust tracking.
Future Improvements 🚀
Add multi-lane tracking for highways.
Implement vehicle speed estimation.
Use DeepSORT for better tracking in crowded scenes.
Deploy as a real-time monitoring system with a web interface.
