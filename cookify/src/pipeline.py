"""
Cookify Pipeline - Main pipeline for recipe extraction from cooking videos
"""

import os
import logging
import json
import yaml
from pathlib import Path
from tqdm import tqdm

from preprocessing.video_processor import VideoProcessor
from frame_analysis.object_detector import ObjectDetector
from frame_analysis.scene_detector import SceneDetector
from frame_analysis.text_recognizer import TextRecognizer
from frame_analysis.action_recognizer import ActionRecognizer
from audio_analysis.transcriber import AudioTranscriber
from audio_analysis.nlp_processor import NLPProcessor
from integration.multimodal_integrator import MultimodalIntegrator
from recipe_extraction.recipe_extractor import RecipeExtractor
from output_formatting.formatter import OutputFormatter
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Main pipeline for recipe extraction from cooking videos.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the pipeline.
        
        Args:
            config_path (str, optional): Path to the configuration file. Defaults to None.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Configure logging
        log_level = getattr(logging, self.config["general"]["log_level"].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("cookify.log"),
                logging.StreamHandler()
            ]
        )
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        # Video preprocessing
        self.video_processor = VideoProcessor(
            output_dir=self.config["general"]["output_dir"],
            frame_rate=self.config["preprocessing"]["frame_rate"],
            temp_dir=self.config["general"]["temp_dir"]
        )
        
        # Frame analysis
        self.object_detector = ObjectDetector(
            confidence_threshold=self.config["object_detection"]["confidence_threshold"]
        )
        
        self.scene_detector = SceneDetector(
            threshold=self.config["scene_detection"]["threshold"],
            min_scene_len=self.config["scene_detection"]["min_scene_len"]
        )
        
        # Initialize other components as needed
        # These will be implemented in their respective modules
        self.text_recognizer = None  # TextRecognizer()
        self.action_recognizer = None  # ActionRecognizer()
        self.audio_transcriber = None  # AudioTranscriber()
        self.nlp_processor = None  # NLPProcessor()
        self.multimodal_integrator = None  # MultimodalIntegrator()
        self.recipe_extractor = None  # RecipeExtractor()
        self.output_formatter = None  # OutputFormatter()
    
    def process(self, video_path, output_path=None):
        """
        Process a video and extract the recipe.
        
        Args:
            video_path (str): Path to the video file.
            output_path (str, optional): Path to save the output. Defaults to None.
            
        Returns:
            dict: Extracted recipe.
        """
        logger.info(f"Processing video: {video_path}")
        
        # Determine output path
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = os.path.join(self.config["general"]["output_dir"], f"{video_name}_recipe.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Step 1: Preprocess video
            logger.info("Preprocessing video...")
            frames, audio_path, metadata = self.video_processor.process(video_path)
            
            # Step 2: Detect scenes
            logger.info("Detecting scenes...")
            scenes = self.scene_detector.detect(video_path)
            
            # Step 3: Detect objects in frames
            logger.info("Detecting objects in frames...")
            object_detections = self.object_detector.detect(frames)
            
            # Filter for cooking-related objects if configured
            if self.config["object_detection"]["filter_cooking_objects"]:
                object_detections = self.object_detector.filter_cooking_related(object_detections)
            
            # Step 4: Recognize text in frames (placeholder)
            logger.info("Recognizing text in frames...")
            text_detections = []  # self.text_recognizer.recognize(frames)
            
            # Step 5: Recognize actions (placeholder)
            logger.info("Recognizing actions...")
            action_detections = []  # self.action_recognizer.recognize(frames, scenes)
            
            # Step 6: Transcribe audio (placeholder)
            logger.info("Transcribing audio...")
            transcription = {}  # self.audio_transcriber.transcribe(audio_path)
            
            # Step 7: Process natural language (placeholder)
            logger.info("Processing natural language...")
            nlp_results = {}  # self.nlp_processor.process(transcription)
            
            # Step 8: Integrate multimodal data (placeholder)
            logger.info("Integrating multimodal data...")
            integrated_data = {
                "video_path": video_path,
                "metadata": metadata,
                "scenes": scenes,
                "object_detections": object_detections,
                "text_detections": text_detections,
                "action_detections": action_detections,
                "transcription": transcription,
                "nlp_results": nlp_results
            }
            
            # Step 9: Extract recipe structure (placeholder)
            logger.info("Extracting recipe structure...")
            recipe = {
                "title": "Sample Recipe",
                "servings": "4",
                "ingredients": [
                    {"name": "ingredient 1", "qty": "1", "unit": "cup"},
                    {"name": "ingredient 2", "qty": "2", "unit": "tbsp"}
                ],
                "tools": ["tool 1", "tool 2"],
                "steps": [
                    {
                        "idx": 1,
                        "start": 0.0,
                        "end": 10.0,
                        "action": "mix",
                        "objects": ["ingredient 1", "ingredient 2"],
                        "details": "Mix ingredients together",
                        "temp": None,
                        "duration": None
                    }
                ]
            }
            
            # Step 10: Format output (placeholder)
            logger.info("Formatting output...")
            formatted_output = recipe
            
            # Save output
            with open(output_path, 'w') as f:
                json.dump(formatted_output, f, indent=2)
            
            logger.info(f"Recipe extracted and saved to {output_path}")
            return formatted_output
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            raise
    
    def print_summary(self, recipe):
        """
        Print a summary of the extracted recipe.
        
        Args:
            recipe (dict): Extracted recipe.
        """
        print("\nRecipe Extraction Summary:")
        print(f"Title: {recipe.get('title', 'Unknown')}")
        print(f"Servings: {recipe.get('servings', 'Unknown')}")
        print(f"Ingredients: {len(recipe.get('ingredients', []))} items")
        print(f"Steps: {len(recipe.get('steps', []))} steps")
