import os
import torch
import ffmpeg
import numpy as np
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline
)
from pydub import AudioSegment
import subprocess
from typing import List, Dict, Tuple, Set
import logging
import json
import cv2
from scipy.signal import find_peaks
import librosa
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class SmartTrailerGenerator:
    def __init__(self, output_dir: str = "output_trailers"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Whisper for dialogue detection
        self.whisper_model = whisper.load_model("base")
        
        # Initialize emotion detection pipeline
        self.emotion_classifier = pipeline("text-classification", 
                                        model="j-hartmann/emotion-english-distilroberta-base",
                                        return_all_scores=True)
        
        # Scene detection parameters
        self.min_scene_length = 1.5  # seconds
        self.max_scene_length = 15   # seconds
        self.target_trailer_length = 120  # seconds
        
        # Genre-specific configurations with emotion mappings
        self.genre_config = {
            "action": {
                "keywords": ["fight", "battle", "explosion", "chase", "gun", "run", "danger"],
                "emotions": ["anger", "fear", "surprise"],
                "visual_cues": ["action scene", "explosion", "fight scene", "chase scene"],
                "audio_features": {
                    "energy_threshold": 0.7,
                    "tempo_min": 120
                },
                "highlight_features": {
                    "motion_threshold": 0.15,
                    "brightness_variance": 0.1,
                    "audio_intensity": 0.7
                }
            },
            "drama": {
                "keywords": ["emotional", "conflict", "relationship", "struggle", "family"],
                "emotions": ["sadness", "neutral", "anger"],
                "visual_cues": ["emotional scene", "dramatic moment", "intense dialogue"],
                "audio_features": {
                    "energy_threshold": 0.4,
                    "speech_rate_threshold": 0.6
                },
                "highlight_features": {
                    "motion_threshold": 0.1,
                    "brightness_variance": 0.15,
                    "dialogue_intensity": 0.6
                }
            },
            "comedy": {
                "keywords": ["funny", "laugh", "joke", "humor", "smile"],
                "emotions": ["joy", "surprise"],
                "visual_cues": ["funny moment", "comedic scene", "laughter"],
                "audio_features": {
                    "laughter_threshold": 0.6,
                    "speech_rate_variance": 0.4
                },
                "highlight_features": {
                    "motion_threshold": 0.2,
                    "brightness_variance": 0.2,
                    "laughter_intensity": 0.5
                }
            },
            "romance": {
                "keywords": ["love", "kiss", "romantic", "date", "relationship"],
                "emotions": ["joy", "love"],
                "visual_cues": ["romantic scene", "intimate moment", "couple"],
                "audio_features": {
                    "music_presence": 0.6,
                    "speech_softness": 0.7
                },
                "highlight_features": {
                    "motion_threshold": 0.08,
                    "brightness_variance": 0.12,
                    "emotional_intensity": 0.6
                }
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video for analysis."""
        audio_path = os.path.join(self.output_dir, "temp_audio.wav")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return audio_path
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            return ""

    def analyze_dialogue(self, audio_path: str) -> List[Dict]:
        """Analyze dialogue using Whisper and emotion detection."""
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(audio_path)
            
            dialogue_segments = []
            for segment in result["segments"]:
                # Get emotion scores for the text
                emotions = self.emotion_classifier(segment["text"])[0]
                
                dialogue_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "emotions": {e["label"]: e["score"] for e in emotions}
                })
            
            return dialogue_segments
        except Exception as e:
            self.logger.error(f"Dialogue analysis failed: {str(e)}")
            return []

    def analyze_audio_features(self, audio_path: str) -> List[Dict]:
        """Analyze audio features using librosa."""
        try:
            y, sr = librosa.load(audio_path)
            
            # Analyze in windows
            hop_length = int(sr * 0.5)  # 0.5 second windows
            
            # Extract features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            audio_features = []
            
            for i in range(0, len(y), hop_length):
                segment = y[i:i + hop_length]
                if len(segment) < hop_length:
                    break
                    
                energy = np.mean(librosa.feature.rms(y=segment))
                
                audio_features.append({
                    "start": i / sr,
                    "end": (i + hop_length) / sr,
                    "energy": float(energy),
                    "tempo": tempo,
                    "spectral_centroid": float(np.mean(spectral_centroids[:, i//hop_length])),
                    "mfcc_mean": float(np.mean(mfcc[:, i//hop_length]))
                })
            
            return audio_features
        except Exception as e:
            self.logger.error(f"Audio feature analysis failed: {str(e)}")
            return []

    def detect_genre_specific_highlights(self, 
                                      video_path: str,
                                      dialogue_segments: List[Dict],
                                      audio_features: List[Dict],
                                      genre: str) -> List[Tuple[float, float, float, str]]:
        """Detect highlights specific to a genre."""
        highlights = []
        genre_config = self.genre_config[genre]
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            window_size = int(fps * 2)  # 2-second windows
            
            # Prepare CLIP for visual analysis
            visual_cues = genre_config["visual_cues"]
            text_inputs = self.clip_processor(text=visual_cues, return_tensors="pt", padding=True)
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % (fps * 2) == 0:  # Analysis every 2 seconds
                    current_time = frame_count / fps
                    
                    # Visual analysis
                    image = self.clip_processor(images=frame, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**image)
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    
                    # Calculate visual similarity scores
                    similarity = torch.nn.functional.cosine_similarity(
                        image_features, text_features
                    )
                    
                    # Get corresponding dialogue and audio features
                    relevant_dialogue = [
                        d for d in dialogue_segments 
                        if d["start"] <= current_time <= d["end"]
                    ]
                    
                    relevant_audio = [
                        a for a in audio_features
                        if a["start"] <= current_time <= a["end"]
                    ]
                    
                    # Calculate genre-specific score
                    score = self.calculate_genre_score(
                        genre,
                        similarity.max().item(),
                        relevant_dialogue,
                        relevant_audio
                    )
                    
                    if score > 0.6:  # Adjustable threshold
                        highlights.append((
                            current_time,
                            current_time + 2,  # 2-second window
                            score,
                            "scene"
                        ))
                
                frame_count += 1
            
            cap.release()
            
            # Add dialogue-based highlights
            for segment in dialogue_segments:
                emotion_score = max(
                    segment["emotions"].get(emotion, 0)
                    for emotion in genre_config["emotions"]
                )
                
                if emotion_score > 0.7:  # Adjustable threshold
                    highlights.append((
                        segment["start"],
                        segment["end"],
                        emotion_score,
                        "dialogue"
                    ))
            
            # Merge and sort highlights
            highlights = self.merge_overlapping_highlights(highlights)
            highlights.sort(key=lambda x: x[2], reverse=True)
            
            return highlights
            
        except Exception as e:
            self.logger.error(f"Genre-specific highlight detection failed: {str(e)}")
            return []

    def calculate_genre_score(self, 
                            genre: str, 
                            visual_score: float,
                            dialogue_segments: List[Dict],
                            audio_features: List[Dict]) -> float:
        """Calculate a genre-specific score based on multiple features."""
        genre_config = self.genre_config[genre]
        
        # Visual score weight
        score = visual_score * 0.4
        
        # Emotion score
        if dialogue_segments:
            emotion_scores = []
            for segment in dialogue_segments:
                relevant_emotions = [
                    segment["emotions"].get(emotion, 0)
                    for emotion in genre_config["emotions"]
                ]
                emotion_scores.append(max(relevant_emotions))
            score += max(emotion_scores) * 0.3
        
        # Audio feature score
        if audio_features:
            audio_score = 0
            for feature in audio_features:
                if genre == "action":
                    audio_score = max(audio_score, 
                                    feature["energy"] * 0.7 + 
                                    (feature["tempo"] / 200) * 0.3)
                elif genre == "drama":
                    audio_score = max(audio_score, 
                                    feature["energy"] * 0.3 + 
                                    feature["mfcc_mean"] * 0.7)
                elif genre == "comedy":
                    audio_score = max(audio_score, 
                                    feature["energy"] * 0.5 + 
                                    feature["spectral_centroid"] * 0.5)
                elif genre == "romance":
                    audio_score = max(audio_score, 
                                    (1 - feature["energy"]) * 0.4 + 
                                    feature["mfcc_mean"] * 0.6)
            
            score += audio_score * 0.3
        
        return score

    def merge_overlapping_highlights(self, 
                                   highlights: List[Tuple[float, float, float, str]]) -> List[Tuple[float, float, float, str]]:
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
                    max(current[2], next_highlight[2]),
                    "combined" if current[3] != next_highlight[3] else current[3]
                )
            else:
                merged.append(current)
                current = next_highlight
        
        merged.append(current)
        return merged

    def create_genre_trailer(self, 
                           video_path: str,
                           highlights: List[Tuple[float, float, float, str]],
                           genre: str) -> str:
        """Create a trailer for a specific genre."""
        try:
            # Select highlights based on target length
            total_duration = 0
            selected_highlights = []
            
            for start, end, score, highlight_type in highlights:
                duration = end - start
                
                # Adjust clip length based on highlight type
                if highlight_type == "dialogue":
                    # Extend dialogue clips slightly
                    start = max(0, start - 0.5)
                    end = end + 0.5
                    duration = end - start
                
                if total_duration + duration <= self.target_trailer_length:
                    selected_highlights.append((start, end, score, highlight_type))
                    total_duration += duration
                
                if total_duration >= self.target_trailer_length:
                    break
            
            # Extract and combine clips
            clip_paths = []
            for i, (start, end, _, _) in enumerate(selected_highlights):
                clip_path = os.path.join(self.output_dir, f"temp_{genre}_clip_{i}.mp4")
                if self.extract_highlight_clip(video_path, start, end, clip_path):
                    clip_paths.append(clip_path)
            
            if clip_paths:
                # Create final trailer
                output_path = os.path.join(self.output_dir, f"{genre}_trailer.mp4")
                
                # Create concat file
                concat_file = os.path.join(self.output_dir, f"concat_{genre}.txt")
                with open(concat_file, 'w') as f:
                    for clip_path in clip_paths:
                        f.write(f"file '{os.path.abspath(clip_path)}'\n")
                
                # Add crossfade transitions
    # Add crossfade transitions
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-filter_complex',
                    '[0:v]fade=t=in:st=0:d=1,fade=t=out:st=' + str(total_duration-1) + ':d=1[v]',
                    '-map', '[v]',
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean up temporary files
                os.remove(concat_file)
                for clip_path in clip_paths:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                
                return output_path
                
        except Exception as e:
            self.logger.error(f"Failed to create {genre} trailer: {str(e)}")
            return ""

    def extract_highlight_clip(self, video_path: str, start: float, end: float, output_path: str) -> bool:
        """Extract a highlight clip with proper audio handling."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', video_path,
                '-t', str(end - start),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-filter:v', 'fade=t=in:st=0:d=0.5,fade=t=out:st=' + str(end-start-0.5) + ':d=0.5',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract clip: {str(e)}")
            return False

    def detect_movie_genres(self, dialogue_segments: List[Dict], audio_features: List[Dict]) -> Set[str]:
        """Detect the primary genres of the movie based on dialogue and audio analysis."""
        genre_scores = defaultdict(float)
        
        # Analyze dialogue emotions
        for segment in dialogue_segments:
            for genre, config in self.genre_config.items():
                # Calculate emotion match
                emotion_score = max(
                    segment["emotions"].get(emotion, 0)
                    for emotion in config["emotions"]
                )
                genre_scores[genre] += emotion_score
                
                # Check for genre keywords
                text = segment["text"].lower()
                keyword_matches = sum(1 for keyword in config["keywords"] if keyword in text)
                genre_scores[genre] += keyword_matches * 0.2
        
        # Analyze audio features
        for feature in audio_features:
            # Action: high energy, fast tempo
            if feature["energy"] > 0.7 and feature["tempo"] > 120:
                genre_scores["action"] += 0.5
            
            # Drama: moderate energy, varied spectral content
            if 0.3 <= feature["energy"] <= 0.6:
                genre_scores["drama"] += 0.3
            
            # Comedy: varied energy, high spectral variance
            if feature["spectral_centroid"] > np.mean([f["spectral_centroid"] for f in audio_features]):
                genre_scores["comedy"] += 0.3
            
            # Romance: smooth audio features, lower energy
            if feature["energy"] < 0.5 and feature["mfcc_mean"] > np.mean([f["mfcc_mean"] for f in audio_features]):
                genre_scores["romance"] += 0.3
        
        # Normalize scores
        total_segments = len(dialogue_segments)
        for genre in genre_scores:
            genre_scores[genre] /= total_segments
        
        # Select genres with scores above threshold
        selected_genres = {
            genre for genre, score in genre_scores.items()
            if score > 0.3  # Adjustable threshold
        }
        
        return selected_genres

    def generate_trailers(self, video_path: str) -> Dict[str, str]:
        """Generate multiple genre-specific trailers based on content analysis."""
        self.logger.info(f"Starting smart trailer generation for {video_path}")
        
        try:
            # Extract audio for analysis
            self.logger.info("Extracting audio...")
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                raise Exception("Failed to extract audio")
            
            # Analyze dialogue and audio
            self.logger.info("Analyzing dialogue and audio...")
            dialogue_segments = self.analyze_dialogue(audio_path)
            audio_features = self.analyze_audio_features(audio_path)
            
            # Detect movie genres
            self.logger.info("Detecting movie genres...")
            detected_genres = self.detect_movie_genres(dialogue_segments, audio_features)
            self.logger.info(f"Detected genres: {detected_genres}")
            
            # Generate trailers for each detected genre
            trailers = {}
            for genre in detected_genres:
                self.logger.info(f"Generating {genre} trailer...")
                highlights = self.detect_genre_specific_highlights(
                    video_path,
                    dialogue_segments,
                    audio_features,
                    genre
                )
                
                if highlights:
                    trailer_path = self.create_genre_trailer(video_path, highlights, genre)
                    if trailer_path:
                        trailers[genre] = trailer_path
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return trailers
            
        except Exception as e:
            self.logger.error(f"Trailer generation failed: {str(e)}")
            return {}

# Usage example
if __name__ == "__main__":
    generator = SmartTrailerGenerator(output_dir="output_trailers")
    video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"  # Replace with your video path
    trailers = generator.generate_trailers(video_path)
    
    print("\nGenerated trailers:")
    for genre, path in trailers.items():
        print(f"{genre}: {path}")