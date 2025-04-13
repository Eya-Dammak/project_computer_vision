import cv2
import numpy as np
from ultralytics import YOLO
import os

class FruitDetector:
    def __init__(self, model_path="yolov8l.pt"):
        # Initialize YOLOv8 model
        self.model = YOLO(model_path)
        
        # Ripeness color thresholds (HSV)
        self.ripeness_thresholds = {
            "apple": {
            "ripe": ([0, 50, 50], [15, 255, 255]),    # Red apples
            "unripe": ([35, 50, 50], [85, 255, 255])  # Green apples
            },
            "banana": {
            "ripe": ([0, 194, 240] ,[0, 206, 255], [57, 218, 255], [109, 227, 255] , [161, 237, 255]),  # Wider yellow range
            "unripe": ([50, 40, 40], [85, 255, 255])  # Green to yellow-green
            },
            "orange": {
            "ripe": ([5, 100, 100], [20, 255, 255]),  # Bright oranges
            "unripe": ([25, 50, 50], [40, 255, 200])  # Greenish oranges
            }


        }

    def detect_fruits(self, img):
        """Detect fruits and estimate ripeness"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = cv2.merge((clahe.apply(l_channel), a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        img = cv2.resize(img, (1280, 1280))

        results = self.model.predict(
        img,
        conf=0.1,  # Lower confidence threshold (default: 0.25)
        iou=0.2,    # Reduced overlap merging (default: 0.7)
        imgsz=1280,  # Optional: Higher resolution (e.g., 1280)
        augment=True  # Test-time augmentation (flips/rotations)
    )
        
        fruit_data = []
        fruit_counts = {}

        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                fruit_type = self.model.names[int(cls)]
                confidence = float(conf)
                
                # Crop fruit region for ripeness analysis
                fruit_roi = img[y1:y2, x1:x2]
                if fruit_roi.size > 0:  # Ensure we have a valid region
                    ripeness = self.estimate_ripeness(fruit_roi, fruit_type)
                else:
                    ripeness = "Unknown"
                
                fruit_data.append({
                    "type": fruit_type,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "ripeness": ripeness
                })
                
                # Update counts
                if fruit_type not in fruit_counts:
                    fruit_counts[fruit_type] = {"count": 0, "ripe": 0, "unripe": 0}
                fruit_counts[fruit_type]["count"] += 1
                if ripeness == "Ripe":
                    fruit_counts[fruit_type]["ripe"] += 1
                elif ripeness == "Unripe":
                    fruit_counts[fruit_type]["unripe"] += 1
                
                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{fruit_type} {confidence:.2f} ({ripeness})"
                cv2.putText(img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        
        return img, fruit_data, fruit_counts

    def estimate_ripeness(self, img, fruit_type):
        """Estimate ripeness using HSV color analysis"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Get thresholds for the fruit type (default to orange if unknown)
        thresholds = self.ripeness_thresholds.get(fruit_type.lower(), 
                        self.ripeness_thresholds["orange"])
        
        # Create masks
        ripe_mask = cv2.inRange(hsv, np.array(thresholds["ripe"][0]), 
                               np.array(thresholds["ripe"][1]))
        unripe_mask = cv2.inRange(hsv, np.array(thresholds["unripe"][0]), 
                                 np.array(thresholds["unripe"][1]))
        
        # Count pixels
        ripe_pixels = cv2.countNonZero(ripe_mask)
        unripe_pixels = cv2.countNonZero(unripe_mask)
        
        # Determine ripeness
        if ripe_pixels > unripe_pixels:
            return "Ripe"
        elif unripe_pixels > ripe_pixels:
            return "Unripe"
        return "Intermediate"

    def process_video(self, video_path, output_path=None):
        """Process video file or stream"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                   (frame_width, frame_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, _, _ = self.detect_fruits(frame)
            
            # Write frame if output specified
            if writer:
                writer.write(processed_frame)
            
            # Display processed frame
            cv2.imshow('Fruit Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()