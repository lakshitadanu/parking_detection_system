import cv2
import json
import torch
import os
from ultralytics import YOLO

# Paths
MODEL_PATH = "best.pt"  # Path to trained YOLOv8 model
VIDEO_PATH = "parking1.mp4"  # Path to recorded parking video
JSON_PATH = "parking_spots.json"  # Path to saved parking spot coordinates

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Load parking spot coordinates
if not os.path.exists(JSON_PATH):
    print(f"Error: {JSON_PATH} not found!")
    exit()

with open(JSON_PATH, "r") as f:
    parking_data = json.load(f)

# Extract parking spots for the given video
video_name = os.path.basename(VIDEO_PATH)
if video_name not in parking_data:
    print(f"Error: No parking spot data found for {video_name} in JSON file.")
    exit()

parking_spots = parking_data[video_name]  # Dictionary of parking spots
TOTAL_SPOTS = len(parking_spots)  # Total number of parking spots

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Unable to open video {VIDEO_PATH}")
    exit()

# Adjustable thresholds
IOU_THRESHOLD = 0.3  # IoU threshold for parking occupancy detection
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for vehicle detection

def calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bounding boxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute intersection
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    # IoU calculation
    return inter_area / union_area if union_area > 0 else 0

def draw_label(img, text, pos, bg_color, text_color=(255, 255, 255)):
    """Draw a label with background rectangle for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = pos
    rect_x1, rect_y1 = text_x, text_y - text_size[1] - 5
    rect_x2, rect_y2 = text_x + text_size[0] + 10, text_y + 5

    # Draw rectangle background
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Draw text with shadow effect
    cv2.putText(img, text, (text_x + 2, text_y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)  # Shadow
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# Create a resizable OpenCV window
cv2.namedWindow("Parking Occupancy Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Parking Occupancy Detection", 1000, 600)

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection on the frame
    results = model(frame)

    # Extract detected vehicles
    vehicle_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            class_id = int(box.cls[0])  # Class ID of detected object
            conf = float(box.conf[0])  # Confidence score

            # Only consider high-confidence vehicle detections
            if conf > CONFIDENCE_THRESHOLD:
                vehicle_boxes.append((x1, y1, x2, y2, class_id, conf))

                # Get class name from YOLO model
                class_name = model.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # Use function for improved label appearance
                draw_label(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), (255, 0, 0))

    # Check if each parking spot is occupied using IoU
    occupied_count = 0  # Count of occupied spots

    for spot_name, coords in parking_spots.items():
        x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
        occupied = False

        for vx1, vy1, vx2, vy2, class_id, conf in vehicle_boxes:
            iou = calculate_iou((x1, y1, x2, y2), (vx1, vy1, vx2, vy2))
            if iou > IOU_THRESHOLD:  # Check IoU threshold
                occupied = True
                occupied_count += 1
                break

        # Define colors for occupied (Red) and empty (Green) spots
        color = (0, 0, 255) if occupied else (0, 255, 0)

        # Draw parking spot bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Use function for improved parking spot label
        draw_label(frame, f"{spot_name}: {'Occupied' if occupied else 'Empty'}", (x1, y1 - 10), color)

    # Display total and occupied parking slots
    info_text = f"Total Spots: {TOTAL_SPOTS} | Occupied: {occupied_count} | Available: {TOTAL_SPOTS - occupied_count}"
    draw_label(frame, info_text, (20, 40), (50, 50, 50), (255, 255, 255))

    # Show the output
    cv2.imshow("Parking Occupancy Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()