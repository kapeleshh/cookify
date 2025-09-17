"""
Object Detector - Detects objects in video frames using YOLOv8
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Cooking-related classes from COCO dataset that YOLOv8 can detect
COOKING_CLASSES = {
    # Food items
    'apple', 'orange', 'banana', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'sandwich',
    
    # Kitchen tools and containers
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    
    # People
    'person',
}

class ObjectDetector:
    """
    Class for detecting objects in video frames using YOLOv8.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize the ObjectDetector.
        
        Args:
            model_path (str, optional): Path to the YOLOv8 model. Defaults to None (uses default model).
            confidence_threshold (float, optional): Confidence threshold for detections. Defaults to 0.25.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Lazy load the model when needed
    
    def _load_model(self):
        """
        Load the YOLOv8 model.
        """
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
            
            # Use specified model path or default to YOLOv8x
            if self.model_path:
                logger.info(f"Loading YOLOv8 model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                logger.info("Loading default YOLOv8x model")
                self.model = YOLO("yolov8x.pt")
            
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect(self, frames):
        """
        Detect objects in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            list: List of detection results for each frame.
                Each result is a dictionary with keys:
                - frame_idx: Index of the frame
                - detections: List of detected objects with class, confidence, and bounding box
        """
        self._load_model()
        
        results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Detecting objects")):
            # Run inference
            yolo_results = self.model(frame, conf=self.confidence_threshold)
            
            # Process results
            frame_results = self._process_results(yolo_results, i)
            results.append(frame_results)
        
        return results
    
    def _process_results(self, yolo_results, frame_idx):
        """
        Process YOLOv8 results into a structured format.
        
        Args:
            yolo_results: Results from YOLOv8 model.
            frame_idx (int): Index of the frame.
            
        Returns:
            dict: Structured detection results.
        """
        detections = []
        
        # Process each detection
        for result in yolo_results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                
                # Get class and confidence
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id]
                confidence = boxes.conf[i].item()
                
                # Filter for cooking-related classes if needed
                # if cls_name.lower() not in COOKING_CLASSES:
                #     continue
                
                # Create detection object
                detection = {
                    "class": cls_name,
                    "confidence": confidence,
                    "box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                
                detections.append(detection)
        
        return {
            "frame_idx": frame_idx,
            "detections": detections
        }
    
    def filter_cooking_related(self, detections):
        """
        Filter detections to only include cooking-related objects.
        
        Args:
            detections (list): List of detection results.
            
        Returns:
            list: Filtered detection results.
        """
        filtered_results = []
        
        for frame_result in detections:
            filtered_detections = [
                d for d in frame_result["detections"]
                if d["class"].lower() in COOKING_CLASSES
            ]
            
            filtered_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": filtered_detections
            })
        
        return filtered_results
