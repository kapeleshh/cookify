"""
Scene Detector - Detects scene changes in videos using SceneDetect
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Class for detecting scene changes in videos using SceneDetect.
    """
    
    def __init__(self, threshold=30.0, min_scene_len=15):
        """
        Initialize the SceneDetector.
        
        Args:
            threshold (float, optional): Threshold for content detection. Defaults to 30.0.
            min_scene_len (int, optional): Minimum scene length in frames. Defaults to 15.
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    def detect(self, video_path):
        """
        Detect scenes in a video.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            list: List of scenes, where each scene is a dictionary with:
                - start_frame: Starting frame number
                - end_frame: Ending frame number
                - start_time: Starting time in seconds
                - end_time: Ending time in seconds
        """
        logger.info(f"Detecting scenes in video: {video_path}")
        
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector
            
            # Open video
            video = open_video(video_path)
            
            # Create scene manager
            scene_manager = SceneManager()
            
            # Add content detector
            scene_manager.add_detector(
                ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
            )
            
            # Perform scene detection
            scene_manager.detect_scenes(video)
            
            # Get scene list and frame metrics
            scene_list = scene_manager.get_scene_list()
            
            # Get fps to calculate timestamps
            fps = video.frame_rate
            
            # Convert scene list to our format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].frame_num
                end_frame = scene[1].frame_num - 1  # Inclusive end frame
                
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                scenes.append({
                    "scene_idx": i,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time
                })
            
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            # Return a single scene covering the entire video as fallback
            return [{
                "scene_idx": 0,
                "start_frame": 0,
                "end_frame": float('inf'),  # Will be replaced with actual frame count later
                "start_time": 0.0,
                "end_time": float('inf'),  # Will be replaced with actual duration later
                "duration": float('inf')  # Will be replaced with actual duration later
            }]
    
    def get_scene_for_frame(self, scenes, frame_idx):
        """
        Get the scene that contains a specific frame.
        
        Args:
            scenes (list): List of scenes.
            frame_idx (int): Frame index.
            
        Returns:
            dict: Scene containing the frame, or None if not found.
        """
        for scene in scenes:
            if scene["start_frame"] <= frame_idx <= scene["end_frame"]:
                return scene
        
        return None
    
    def get_scene_for_timestamp(self, scenes, timestamp):
        """
        Get the scene that contains a specific timestamp.
        
        Args:
            scenes (list): List of scenes.
            timestamp (float): Timestamp in seconds.
            
        Returns:
            dict: Scene containing the timestamp, or None if not found.
        """
        for scene in scenes:
            if scene["start_time"] <= timestamp <= scene["end_time"]:
                return scene
        
        return None
