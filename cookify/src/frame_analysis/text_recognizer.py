"""
Text Recognizer - Recognizes text in video frames using OCR
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TextRecognizer:
    """
    Class for recognizing text in video frames using OCR.
    """
    
    def __init__(self, languages=None, confidence_threshold=0.5, enhance_text=True):
        """
        Initialize the TextRecognizer.
        
        Args:
            languages (list, optional): List of languages to detect. Defaults to ['en'].
            confidence_threshold (float, optional): Confidence threshold for text detection. Defaults to 0.5.
            enhance_text (bool, optional): Whether to enhance text regions before OCR. Defaults to True.
        """
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
        self.enhance_text = enhance_text
        self.reader = None
        
        # Lazy load the OCR model when needed
    
    def _load_model(self):
        """
        Load the OCR model.
        """
        if self.reader is not None:
            return
        
        try:
            import easyocr
            
            logger.info(f"Loading EasyOCR model for languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages)
            logger.info("EasyOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading EasyOCR model: {e}")
            raise
    
    def recognize(self, frames):
        """
        Recognize text in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            list: List of text detection results for each frame.
                Each result is a dictionary with keys:
                - frame_idx: Index of the frame
                - detections: List of detected text with text, confidence, and bounding box
        """
        self._load_model()
        
        results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Recognizing text")):
            # Enhance text regions if configured
            if self.enhance_text:
                frame = self._enhance_text_regions(frame)
            
            # Run OCR
            ocr_results = self.reader.readtext(frame)
            
            # Process results
            frame_results = self._process_results(ocr_results, i)
            results.append(frame_results)
        
        return results
    
    def _enhance_text_regions(self, frame):
        """
        Enhance text regions in a frame to improve OCR accuracy.
        
        Args:
            frame (numpy.ndarray): Input frame.
            
        Returns:
            numpy.ndarray: Enhanced frame.
        """
        try:
            import cv2
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to remove noise
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Convert back to BGR for OCR
            enhanced = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Error enhancing text regions: {e}")
            return frame
    
    def _process_results(self, ocr_results, frame_idx):
        """
        Process OCR results into a structured format.
        
        Args:
            ocr_results: Results from EasyOCR.
            frame_idx (int): Index of the frame.
            
        Returns:
            dict: Structured text detection results.
        """
        detections = []
        
        # Process each detection
        for result in ocr_results:
            # Get bounding box, text, and confidence
            box = result[0]
            text = result[1]
            confidence = result[2]
            
            # Filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Create detection object
            detection = {
                "text": text,
                "confidence": confidence,
                "box": {
                    "top_left": (float(box[0][0]), float(box[0][1])),
                    "top_right": (float(box[1][0]), float(box[1][1])),
                    "bottom_right": (float(box[2][0]), float(box[2][1])),
                    "bottom_left": (float(box[3][0]), float(box[3][1]))
                }
            }
            
            detections.append(detection)
        
        return {
            "frame_idx": frame_idx,
            "detections": detections
        }
    
    def filter_cooking_related(self, detections, cooking_keywords=None):
        """
        Filter text detections to only include cooking-related text.
        
        Args:
            detections (list): List of text detection results.
            cooking_keywords (list, optional): List of cooking-related keywords.
                Defaults to a predefined list.
            
        Returns:
            list: Filtered text detection results.
        """
        if cooking_keywords is None:
            cooking_keywords = [
                "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
                "gram", "kg", "ml", "liter", "minute", "hour", "temperature", "degrees",
                "celsius", "fahrenheit", "bake", "boil", "simmer", "fry", "roast", "grill",
                "chop", "slice", "dice", "mince", "mix", "stir", "blend", "whisk", "fold",
                "heat", "cook", "recipe", "ingredient", "serving"
            ]
        
        filtered_results = []
        
        for frame_result in detections:
            filtered_detections = []
            
            for detection in frame_result["detections"]:
                text = detection["text"].lower()
                
                # Check if any cooking keyword is in the text
                if any(keyword in text for keyword in cooking_keywords):
                    filtered_detections.append(detection)
            
            filtered_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": filtered_detections
            })
        
        return filtered_results
