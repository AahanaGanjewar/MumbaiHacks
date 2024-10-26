import os
import torch
import ffmpeg
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from pydub import AudioSegment
import subprocess
from typing import List, Dict, Tuple
import logging
import json
import cv2
from scipy.signal import find_peaks

class MovieTrailerGenerator:
    def __init__(self, output_dir: str = "output_trailers"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Adjust scene detection parameters
        self.min_scene_length = 2  # seconds
        self.max_scene_length = 8  # seconds
        self.target_trailer_length = 60  # seconds
        
        # Lower thresholds for more lenient highlight detection
        self.motion_threshold = 0.15  # Lowered from 0.4
        self.brightness_variance_threshold = 0.1  # Lowered from 0.15
        
        self.genre_config = {
            "action": {
                "keywords": ["fight", "battle", "explosion", "chase", "gun", "run"],
                "visual_cues": ["action scene", "explosion", "fight scene", "chase scene"],
                "highlight_features": {
                    "motion_threshold": 0.08,
                    "brightness_variance": 0.08,
                    "audio_intensity": 0.08
                }
            },
            "drama": {
                "keywords": ["emotional", "conflict", "relationship", "struggle"],
                "visual_cues": ["emotional scene", "dramatic moment", "intense dialogue"],
                "highlight_features": {
                    "motion_threshold": 0.15,
                    "brightness_variance": 0.12,
                    "audio_intensity": 0.2
                }
            },
            "comedy": {
                "keywords": ["funny", "laugh", "joke", "humor"],
                "visual_cues": ["funny moment", "comedic scene", "laughter"],
                "highlight_features": {
                    "motion_threshold": 0.2,
                    "brightness_variance": 0.15,
                    "audio_intensity": 0.25
                }
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_highlights(self, video_path: str) -> List[Tuple[float, float, float]]:
        """
        Detect highlight moments in the video based on motion, audio, and visual features.
        Returns list of (start_time, end_time, intensity_score) tuples.
        """
        highlights = []
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Error: Could not open video file")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
            
            # Parameters for highlight detection
            window_size = int(fps * 2)  # 2-second windows
            motion_scores = []
            brightness_scores = []
            
            prev_frame = None
            frame_count = 0
            processed_frames = 0
            
            # Process every nth frame to improve performance
            frame_skip = 2
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate motion score
                if prev_frame is not None:
                    motion = cv2.absdiff(gray, prev_frame)
                    motion_score = np.mean(motion) / 255.0
                    motion_scores.append(motion_score)
                    
                    # Debug logging for motion detection
                    if processed_frames % 100 == 0:
                        self.logger.debug(f"Frame {frame_count}: Motion score = {motion_score:.4f}")
                
                # Calculate brightness variance
                brightness = np.mean(gray) / 255.0
                brightness_scores.append(brightness)
                
                prev_frame = gray.copy()
                frame_count += 1
                processed_frames += 1
                
                # Process in windows
                if len(motion_scores) >= window_size:
                    window_motion = np.mean(motion_scores[-window_size:])
                    window_brightness_var = np.var(brightness_scores[-window_size:])
                    
                    # Debug logging for window processing
                    if processed_frames % 100 == 0:
                        self.logger.debug(f"Window stats - Motion: {window_motion:.4f}, Brightness var: {window_brightness_var:.4f}")
                    
                    # Detect highlight moments with lower thresholds
                    if (window_motion > self.motion_threshold or 
                        window_brightness_var > self.brightness_variance_threshold):
                        start_time = (frame_count - window_size * frame_skip) / fps
                        end_time = frame_count / fps
                        intensity = (window_motion + window_brightness_var) / 2
                        highlights.append((start_time, end_time, intensity))
                        self.logger.info(f"Highlight detected at {start_time:.2f}s - {end_time:.2f}s (intensity: {intensity:.4f})")
                    
                    # Clear old scores
                    if len(motion_scores) > window_size * 2:
                        motion_scores = motion_scores[-window_size:]
                        brightness_scores = brightness_scores[-window_size:]
            
            cap.release()
            
            if not highlights:
                self.logger.warning("No highlights detected with current thresholds")
                return []
            
            # Merge overlapping highlights
            highlights = self.merge_overlapping_highlights(highlights)
            
            # Sort by intensity
            highlights.sort(key=lambda x: x[2], reverse=True)
            
            self.logger.info(f"Final number of highlights detected: {len(highlights)}")
            return highlights
            
        except Exception as e:
            self.logger.error(f"Highlight detection failed: {str(e)}")
            return []

    def merge_overlapping_highlights(self, highlights: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Merge overlapping highlight segments."""
        if not highlights:
            return []
        
        # Sort by start time
        highlights.sort(key=lambda x: x[0])
        
        merged = []
        current = highlights[0]
        
        for next_highlight in highlights[1:]:
            if current[1] >= next_highlight[0]:
                # Merge overlapping segments
                current = (
                    current[0],
                    max(current[1], next_highlight[1]),
                    max(current[2], next_highlight[2])
                )
            else:
                merged.append(current)
                current = next_highlight
        
        merged.append(current)
        return merged

    def extract_highlight_clip(self, video_path: str, start: float, end: float, output_path: str):
        """Extract a highlight clip from the video."""
        try:
            duration = end - start
            if duration > self.max_scene_length:
                # Center the clip if it's too long
                center = (start + end) / 2
                start = center - (self.max_scene_length / 2)
                end = center + (self.max_scene_length / 2)
            
            # Use slower but more accurate seeking
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', video_path,
                '-t', str(end - start),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"Successfully extracted clip: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract clip: {str(e)}")
            return False

    def create_trailer(self, video_path: str, highlights: List[Tuple[float, float, float]], 
                      genre: str) -> str:
        """Create a trailer from selected highlights."""
        try:
            # Select top highlights that fit within target duration
            total_duration = 0
            selected_highlights = []
            
            for start, end, intensity in highlights:
                duration = end - start
                if total_duration + duration <= self.target_trailer_length:
                    selected_highlights.append((start, end, intensity))
                    total_duration += duration
                    self.logger.info(f"Selected highlight: {start:.2f}s - {end:.2f}s (intensity: {intensity:.4f})")
                    
                    if total_duration >= self.target_trailer_length:
                        break
            
            self.logger.info(f"Total selected highlights: {len(selected_highlights)}")
            
            # Extract individual highlight clips
            clip_paths = []
            for i, (start, end, _) in enumerate(selected_highlights):
                clip_path = os.path.join(self.output_dir, f"temp_highlight_{genre}_{i}.mp4")
                if self.extract_highlight_clip(video_path, start, end, clip_path):
                    clip_paths.append(clip_path)
            
            # Concatenate highlights into final trailer
            if clip_paths:
                output_path = os.path.join(self.output_dir, f"{genre}_trailer.mp4")
                
                # Create concat file
                concat_file = os.path.join(self.output_dir, "concat_list.txt")
                with open(concat_file, 'w') as f:
                    for clip_path in clip_paths:
                        f.write(f"file '{os.path.abspath(clip_path)}'\n")
                
                # Concatenate clips
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                self.logger.info(f"Successfully created trailer: {output_path}")
                
                # Clean up temporary files
                os.remove(concat_file)
                for clip_path in clip_paths:
                    os.remove(clip_path)
                
                return output_path
                
        except Exception as e:
            self.logger.error(f"Failed to create trailer: {str(e)}")
            return ""

    def generate_trailers(self, video_path: str) -> Dict[str, str]:
        """Generate trailers for different genres."""
        self.logger.info(f"Starting trailer generation for {video_path}")
        
        try:
            # Verify video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Detect highlights
            self.logger.info("Detecting highlights...")
            highlights = self.detect_highlights(video_path)
            
            if not highlights:
                self.logger.error("No highlights detected")
                return {}
            
            self.logger.info(f"Detected {len(highlights)} potential highlights")
            
            # Create genre-specific trailers
            trailers = {}
            for genre in self.genre_config.keys():
                self.logger.info(f"Creating {genre} trailer...")
                trailer_path = self.create_trailer(video_path, highlights, genre)
                if trailer_path:
                    trailers[genre] = trailer_path
            
            return trailers
            
        except Exception as e:
            self.logger.error(f"Trailer generation failed: {str(e)}")
            return {}

# Usage example
if __name__ == "__main__":
    generator = MovieTrailerGenerator(output_dir="output_trailers")
    video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"  # Replace with your video path
    trailers = generator.generate_trailers(video_path)
    
    print("\nGenerated trailers:")
    for genre, path in trailers.items():
        print(f"{genre}: {path}")