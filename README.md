Vehicle Tracking and Counting Using YOLOv8 and SORT
This project implements real-time vehicle detection, tracking, and counting using YOLOv8 and SORT (Simple Online and Realtime Tracker). It processes a video feed to identify and track vehicles crossing a predefined counting line.

Features 🚗🚛🚌🏍️
Detects Vehicles: Detects and tracks cars, trucks, buses, and motorbikes.
Unique Vehicle Tracking: Assigns a unique ID to each detected vehicle.
Vehicle Counting: Counts vehicles that cross a defined line.
Overlays Information: Displays bounding boxes, object IDs, and the total count.
Uses a Custom Mask: Applies a mask for specific regions (optional).
Requirements 📌
Install the dependencies before running the script:

bash
Copy
Edit
pip install ultralytics opencv-python cvzone numpy sort
How to Run 🏃‍♂️
Place Your Video File in the ../Videos/ directory (modify the path if needed).
Ensure YOLO Weights Are Available in ../Yolo-Weights/yolov8l.pt.
Optional: Use a custom mask.png to track specific regions.
Run the Script:
bash
Copy
Edit
python vehicle_tracking.py
Press 'Q' to Exit the video playback.
Configuration ⚙️
Change the Tracking Line: Modify limits = [400, 297, 673, 297] to set the counting line.
Adjust Confidence Threshold: Change conf > 0.3 for stricter or looser detections.
Modify Classes to Track: Edit if currentClass in ["car", "truck", "bus", "motorbike"] to track other objects.
Notes 📝
Works best with a top-down camera angle for accurate tracking.
If detection is inaccurate, try fine-tuning YOLOv8 on a custom dataset.
The SORT tracker may lose track if vehicles overlap too much.
Example Output 📸
The script will display:

Vehicles with bounding boxes and unique IDs.
A red counting line that turns green when a vehicle crosses.
The total count of unique vehicles displayed on the screen.
Future Improvements 🚀
Add support for multi-lane tracking.
Improve tracking with DeepSORT instead of SORT.
Implement speed estimation for vehicles.
