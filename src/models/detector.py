import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

class LayoutDetector:
    def __init__(self, model_path: str = "models/best.pt"):
        """
        Initialize the detector with YOLO model and QR code detector
        
        Args:
            model_path: Path to the trained YOLO model
        """
        self.yolo_model = YOLO(model_path)
        self.qr_detector = cv2.QRCodeDetector()
        
    def _check_overlap(self, box1: List[float], box2: List[float], threshold: float = 0.2) -> bool:
        """
        Check if two bounding boxes overlap significantly
        
        Args:
            box1: First box coordinates [x1, y1, x2, y2]
            box2: Second box coordinates [x1, y1, x2, y2]
            threshold: IoU threshold for significant overlap (lowered to 0.2 to be more strict)
            
        Returns:
            bool: True if boxes overlap significantly
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou > threshold
        
    def detect_all(self, image_input: Union[str, np.ndarray, Path]) -> Dict[str, List[Dict[str, any]]]:
        """
        Detect faces, logos and QR codes in the image
        
        Args:
            image_input: Path to the image file or numpy array containing the image
            
        Returns:
            Dictionary containing detected objects and their coordinates
        """
        # Load image if path is provided
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Could not load image from {image_input}")
        else:
            image = image_input
            
        results = {
            "faces": [],
            "logos": [],
            "qrcodes": []
        }
        
        # Detect QR codes using OpenCV QR detector
        qr_boxes = []
        retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(image)
        if retval:
            for qr_points, qr_data in zip(points, decoded_info):
                if qr_points is not None:
                    x_coords = qr_points[:, 0]
                    y_coords = qr_points[:, 1]
                    x1, y1 = float(min(x_coords)), float(min(y_coords))
                    x2, y2 = float(max(x_coords)), float(max(y_coords))
                    
                    # Calculate box dimensions
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Add significant padding (50% of width/height) to QR box
                    padding_x = width * 0.5
                    padding_y = height * 0.5
                    
                    x1_padded = max(0, x1 - padding_x)
                    y1_padded = max(0, y1 - padding_y)
                    x2_padded = min(image.shape[1], x2 + padding_x)
                    y2_padded = min(image.shape[0], y2 + padding_y)
                    
                    # Store original box for display
                    original_box = [x1, y1, x2, y2]
                    
                    # Store padded box for overlap checking
                    padded_box = [x1_padded, y1_padded, x2_padded, y2_padded]
                    qr_boxes.append(padded_box)
                    
                    results["qrcodes"].append({
                        "bbox": original_box,
                        "data": qr_data if qr_data else "QR Code detected"
                    })
        
        # Then detect faces and logos using YOLO
        yolo_results = self.yolo_model(image)[0]
        for detection in yolo_results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            obj_type = yolo_results.names[int(cls)]
            
            if obj_type in ["face", "logo"]:
                # Convert to list of floats
                current_box = [float(x1), float(y1), float(x2), float(y2)]
                
                # Check if this detection overlaps with any QR code's padded area
                overlaps_with_qr = any(
                    self._check_overlap(current_box, qr_box) 
                    for qr_box in qr_boxes
                )
                
                # Only add detection if it doesn't overlap with QR codes
                if not overlaps_with_qr:
                    results[f"{obj_type}s"].append({
                        "bbox": current_box,
                        "confidence": float(conf)
                    })
        
        return results