"""
Action Recognizer - Recognizes cooking actions in video frames
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Common cooking actions that can be recognized
COOKING_ACTIONS = {
    "cutting": ["cutting", "chopping", "slicing", "dicing", "mincing"],
    "mixing": ["mixing", "stirring", "whisking", "blending", "folding"],
    "frying": ["frying", "sautéing", "searing"],
    "boiling": ["boiling", "simmering", "poaching"],
    "baking": ["baking", "roasting"],
    "grilling": ["grilling", "broiling"],
    "measuring": ["measuring", "weighing", "pouring"],
    "kneading": ["kneading", "rolling", "shaping"],
    "plating": ["plating", "garnishing", "serving"]
}

class ActionRecognizer:
    """
    Class for recognizing cooking actions in video frames.
    """
    
    def __init__(self, confidence_threshold=0.5, frame_window=16, use_optical_flow=False):
        """
        Initialize the ActionRecognizer.
        
        Args:
            confidence_threshold (float, optional): Confidence threshold for action recognition. Defaults to 0.5.
            frame_window (int, optional): Number of frames to use for action recognition. Defaults to 16.
            use_optical_flow (bool, optional): Whether to use optical flow for action recognition. Defaults to False.
        """
        self.confidence_threshold = confidence_threshold
        self.frame_window = frame_window
        self.use_optical_flow = use_optical_flow
        self.model = None
        
        # Lazy load the model when needed
    
    def _load_model(self):
        """
        Load the action recognition model.
        """
        if self.model is not None:
            return
        
        try:
            # This is a placeholder for loading the MMAction2 model
            # In a real implementation, we would load the model here
            logger.info("Loading action recognition model...")
            self.model = "placeholder_model"
            logger.info("Action recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading action recognition model: {e}")
            raise
    
    def recognize(self, frames, scenes=None):
        """
        Recognize actions in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            scenes (list, optional): List of scenes. If provided, actions will be recognized per scene.
            
        Returns:
            list: List of action recognition results.
                Each result is a dictionary with keys:
                - start_frame: Starting frame number
                - end_frame: Ending frame number
                - start_time: Starting time in seconds
                - end_time: Ending time in seconds
                - action: Recognized action
                - confidence: Confidence score
        """
        self._load_model()
        
        # If no scenes are provided, treat the entire video as one scene
        if scenes is None:
            scenes = [{
                "scene_idx": 0,
                "start_frame": 0,
                "end_frame": len(frames) - 1,
                "start_time": 0.0,
                "end_time": len(frames) / 30.0,  # Assuming 30 fps
                "duration": len(frames) / 30.0
            }]
        
        results = []
        
        # Process each scene
        for scene in tqdm(scenes, desc="Recognizing actions"):
            # Get frames for this scene
            scene_start = max(0, scene["start_frame"])
            scene_end = min(len(frames) - 1, scene["end_frame"])
            
            # Skip if scene is out of bounds
            if scene_start >= len(frames) or scene_end < 0 or scene_start > scene_end:
                continue
            
            scene_frames = frames[scene_start:scene_end+1]
            
            # Skip if not enough frames
            if len(scene_frames) < self.frame_window:
                continue
            
            # Sample frames at regular intervals to match frame_window
            sampled_indices = np.linspace(0, len(scene_frames) - 1, self.frame_window, dtype=int)
            sampled_frames = [scene_frames[i] for i in sampled_indices]
            
            # Compute optical flow if configured
            if self.use_optical_flow:
                flow_frames = self._compute_optical_flow(sampled_frames)
                # In a real implementation, we would use the flow frames for action recognition
            
            # Recognize action
            action, confidence = self._recognize_action(sampled_frames)
            
            # Skip if confidence is below threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Create result
            result = {
                "start_frame": scene_start,
                "end_frame": scene_end,
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "action": action,
                "confidence": confidence
            }
            
            results.append(result)
        
        logger.info(f"Recognized {len(results)} actions")
        return results
    
    def _compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames.
        
        Args:
            frames (list): List of frames.
            
        Returns:
            list: List of optical flow frames.
        """
        try:
            import cv2
            
            flow_frames = []
            
            for i in range(1, len(frames)):
                # Convert frames to grayscale
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Convert flow to RGB
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(frames[i])
                hsv[..., 1] = 255
                hsv[..., 0] = angle * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                flow_frames.append(flow_rgb)
            
            return flow_frames
        except Exception as e:
            logger.warning(f"Error computing optical flow: {e}")
            return []
    
    def _recognize_action(self, frames):
        """
        Recognize action in a sequence of frames.
        
        Args:
            frames (list): List of frames.
            
        Returns:
            tuple: (action, confidence)
        """
        # This is a placeholder implementation
        # In a real implementation, we would use the MMAction2 model to recognize actions
        
        # For now, return a random cooking action with a random confidence
        import random
        
        action_categories = list(COOKING_ACTIONS.keys())
        action = random.choice(action_categories)
        confidence = random.uniform(0.5, 1.0)
        
        return action, confidence
    
    def map_to_cooking_action(self, action):
        """
        Map a general action to a cooking-specific action.
        
        Args:
            action (str): General action.
            
        Returns:
            str: Cooking-specific action.
        """
        action_lower = action.lower()
        
        for cooking_action, synonyms in COOKING_ACTIONS.items():
            if action_lower in synonyms or action_lower == cooking_action:
                return cooking_action
        
        return action
