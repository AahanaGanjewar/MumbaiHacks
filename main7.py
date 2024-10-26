import os
import torch
import ffmpeg
import numpy as np
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    pipeline
)
import subprocess
from typing import List, Dict, Tuple, Set
import logging
import json
import cv2
import librosa
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class SmartTrailerGenerator:
    def __init__(self, output_dir = "output_trailers"):
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
        self.max_scene_length = 8    # seconds
        self.target_trailer_length = 120  # seconds
        
        # Genre-specific configurations
        self.genre_config = {
            "action": {
                "keywords": ["fight", "battle", "explosion", "chase", "gun", "run", "danger"],
                "emotions": ["anger", "fear", "surprise"],
                "visual_cues": ["action scene", "explosion", "fight scene", "chase scene"],
                "audio_features": {
                    "energy_threshold": 0.7,
                    "tempo_min": 120
                },
                "cinematic_effects": {
                    "color_grade": "high_contrast",
                    "speed": "dynamic",
                    "transition": "fast_cut",
                    "audio_effects": ["impact_sound", "bass_boost"],
                    "score_type": "intense"
                },
                "threshold": 0.65
            },
            "drama": {
                "keywords": ["emotional", "conflict", "relationship", "struggle", "family"],
                "emotions": ["sadness", "neutral", "anger"],
                "visual_cues": ["emotional scene", "dramatic moment", "intense dialogue"],
                "audio_features": {
                    "energy_threshold": 0.4,
                    "speech_rate_threshold": 0.6
                },
                "cinematic_effects": {
                    "color_grade": "muted",
                    "speed": "normal",
                    "transition": "crossfade",
                    "audio_effects": ["reverb", "ambient_pad"],
                    "score_type": "emotional"
                },
                "threshold": 0.7
            },
            "comedy": {
                "keywords": ["funny", "laugh", "joke", "humor", "smile"],
                "emotions": ["joy", "surprise"],
                "visual_cues": ["funny moment", "comedic scene", "laughter"],
                "audio_features": {
                    "laughter_threshold": 0.6,
                    "speech_rate_variance": 0.4
                },
                "cinematic_effects": {
                    "color_grade": "vibrant",
                    "speed": "varied",
                    "transition": "whip_pan",
                    "audio_effects": ["quirky_sound", "upbeat"],
                    "score_type": "light"
                },
                "threshold": 0.75  # Higher threshold for comedy
            },
            "romance": {
                "keywords": ["love", "kiss", "romantic", "date", "relationship"],
                "emotions": ["joy", "love"],
                "visual_cues": ["romantic scene", "intimate moment", "couple"],
                "audio_features": {
                    "music_presence": 0.6,
                    "speech_softness": 0.7
                },
                "cinematic_effects": {
                    "color_grade": "warm",
                    "speed": "slow",
                    "transition": "dissolve",
                    "audio_effects": ["soft_piano", "string_pad"],
                    "score_type": "romantic"
                },
                "threshold": 0.7
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video for analysis."""
        audio_path = os.path.join(self.output_dir, r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\audio.wav")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',  # Higher sample rate for better quality
                '-ac', '2',      # Stereo audio
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
            result = self.whisper_model.transcribe(audio_path)
            
            dialogue_segments = []
            for segment in result["segments"]:
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
            
            hop_length = int(sr * 0.5)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Additional audio features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            
            audio_features = []
            
            for i in range(0, len(y), hop_length):
                segment = y[i:i + hop_length]
                if len(segment) < hop_length:
                    break
                    
                energy = np.mean(librosa.feature.rms(y=segment))
                onset_strength = np.mean(onset_env[i//hop_length:(i+hop_length)//hop_length])
                
                audio_features.append({
                    "start": i / sr,
                    "end": (i + hop_length) / sr,
                    "energy": float(energy),
                    "tempo": tempo,
                    "spectral_centroid": float(np.mean(spectral_centroids[:, i//hop_length])),
                    "mfcc_mean": float(np.mean(mfcc[:, i//hop_length])),
                    "onset_strength": float(onset_strength),
                    "pulse": float(np.mean(pulse[i//hop_length:(i+hop_length)//hop_length]))
                })
            
            return audio_features
        except Exception as e:
            self.logger.error(f"Audio feature analysis failed: {str(e)}")
            return []

    def apply_cinematic_effects(self, clip_path: str, genre: str, output_path: str) -> bool:
        """Apply genre-specific cinematic effects to video clip."""
        try:
            effects = self.genre_config[genre]["cinematic_effects"]
            
            # Build complex filter for combining effects
            filter_complex = []
            
            # Color grading
            if effects["color_grade"] == "high_contrast":
                filter_complex.append("eq=contrast=1.3:saturation=1.2:brightness=0.1")
            elif effects["color_grade"] == "muted":
                filter_complex.append("eq=contrast=0.9:saturation=0.8")
            elif effects["color_grade"] == "vibrant":
                filter_complex.append("eq=contrast=1.1:saturation=1.3")
            elif effects["color_grade"] == "warm":
                filter_complex.append("eq=contrast=1.1:saturation=1.1:gamma_r=1.1:gamma_b=0.9")

            # Speed effects
            if effects["speed"] == "dynamic":
                filter_complex.append("setpts=0.8*PTS")  # Slightly faster
            elif effects["speed"] == "slow":
                filter_complex.append("setpts=1.2*PTS")  # Slightly slower
            
            # Audio effects
            audio_filter = []
            if "impact_sound" in effects["audio_effects"]:
                audio_filter.append("acompressor=threshold=0.125:ratio=20:attack=5:release=50")
                audio_filter.append("bass=g=3:frequency=100:width_type=h")
            elif "reverb" in effects["audio_effects"]:
                audio_filter.append("aecho=0.8:0.8:40|50|70:0.4|0.3|0.2")
            elif "soft_piano" in effects["audio_effects"]:
                audio_filter.append("lowpass=f=3000,highpass=f=200")
                audio_filter.append("acompressor=threshold=0.1:ratio=2:attack=5:release=50")

            # Combine filters
            video_filter = ','.join(filter_complex)
            audio_filter = ','.join(audio_filter)

            cmd = [
                'ffmpeg', '-y',
                '-i', clip_path,
                '-filter_complex', 
                f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]",
                '-map', '[v]', 
                '-map', '[a]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',  # Higher quality
                '-c:a', 'aac',
                '-b:a', '192k',  # Higher audio bitrate
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply cinematic effects: {str(e)}")
            return False

    def detect_genre_specific_highlights(self, 
                                      video_path: str,
                                      dialogue_segments: List[Dict],
                                      audio_features: List[Dict],
                                      genre: str) -> List[Tuple[float, float, float, str]]:
        """Detect highlights specific to a genre with improved scoring."""
        highlights = []
        genre_config = self.genre_config[genre]
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            window_size = int(fps * 2)
            
            visual_cues = genre_config["visual_cues"]
            text_inputs = self.clip_processor(text=visual_cues, return_tensors="pt", padding=True)
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % window_size == 0:
                    current_time = frame_count / fps
                    
                    # Visual analysis
                    image = self.clip_processor(images=frame, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**image)
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    
                    similarity = torch.nn.functional.cosine_similarity(
                        image_features, text_features
                    )
                    
                    # Enhanced context analysis
                    relevant_dialogue = [
                        d for d in dialogue_segments 
                        if d["start"] <= current_time <= d["end"]
                    ]
                    
                    relevant_audio = [
                        a for a in audio_features
                        if a["start"] <= current_time <= a["end"]
                    ]
                    
                    score = self.calculate_genre_score(
                        genre,
                        similarity.max().item(),
                        relevant_dialogue,
                        relevant_audio
                    )
                    
                    # Use genre-specific threshold
                    if score > genre_config["threshold"]:
                        highlights.append((
                            current_time,
                            current_time + self.min_scene_length,
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
                
                if emotion_score > genre_config["threshold"]:
                    highlights.append((
                        segment["start"],
                        segment["end"],
                        emotion_score,
                        "dialogue"
                    ))
            
            highlights = self.merge_overlapping_highlights(highlights)
            highlights.sort(key=lambda x: x[2], reverse=True)
            
            return highlights
            
        except Exception as e:
            self.logger.error(f"Genre-specific highlight detection failed: {str(e)}")
            return []

    def create_genre_trailer(self, 
                           video_path: str,
                           highlights: List[Tuple[float, float, float, str]],
                           genre: str) -> str:
        """Create a trailer with enhanced audio handling and cinematic effects."""
        try:
            # Select highlights based on target length
            total_duration = 0
            selected_highlights = []
            
            for start, end, score, highlight_type in highlights:
                duration = end - start
                
                if highlight_type == "dialogue":
                    start = max(0, start - 0.5)
                    end = end + 0.5
                    duration = end - start
                
                if total_duration + duration <= self.target_trailer_length:
                    selected_highlights.append((start, end, score, highlight_type))
                    total_duration += duration
                
                if total_duration >= self.target_trailer_length:
                    break
            
            # Extract and process clips
            processed_clips = []
            for i, (start, end, _, _) in enumerate(selected_highlights):
                # Extract raw clip
              


                    # Continuing from create_genre_trailer method...
                temp_clip = os.path.join(self.output_dir, f"temp_raw_{genre}clip{i}.mp4")
                processed_clip = os.path.join(self.output_dir, f"temp_processed_{genre}clip{i}.mp4")
                
                # Extract clip with proper audio
                if self.extract_highlight_clip(video_path, start, end, temp_clip):
                    # Apply cinematic effects
                    if self.apply_cinematic_effects(temp_clip, genre, processed_clip):
                        processed_clips.append(processed_clip)
                    
                    # Clean up raw clip
                    if os.path.exists(temp_clip):
                        os.remove(temp_clip)
            
            if processed_clips:
                # Create final trailer
                output_path = os.path.join(self.output_dir, f"{genre}_trailer.mp4")
                
                # Create concat file
                concat_file = os.path.join(self.output_dir, f"concat_{genre}.txt")
                with open(concat_file, 'w') as f:
                    for clip_path in processed_clips:
                        f.write(f"file '{os.path.abspath(clip_path)}'\n")
                
                # Complex filter for transitions and audio mixing
                filter_complex = []
                
                # Video transitions based on genre
                genre_effects = self.genre_config[genre]["cinematic_effects"]
                transition_type = genre_effects["transition"]
                
                if transition_type == "fast_cut":
                    transition_duration = 0.2
                elif transition_type in ["crossfade", "dissolve"]:
                    transition_duration = 0.5
                else:
                    transition_duration = 0.3
                
                # Build ffmpeg command with enhanced audio handling
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-filter_complex',
                    f'[0:v]fade=t=in:st=0:d=1,fade=t=out:st={total_duration-1}:d=1[v];' +
                    f'[0:a]afade=t=in:st=0:d=1,afade=t=out:st={total_duration-1}:d=1[a]',
                    '-map', '[v]',
                    '-map', '[a]',
                    '-c:v', 'libx264',
                    '-preset', 'slow',  # Better quality
                    '-crf', '18',       # Higher quality
                    '-c:a', 'aac',
                    '-b:a', '192k',     # Higher audio bitrate
                    '-ar', '48000',     # Higher audio sample rate
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean up temporary files
                os.remove(concat_file)
                for clip_path in processed_clips:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                
                return output_path
                
        except Exception as e:
            self.logger.error(f"Failed to create {genre} trailer: {str(e)}")
            return ""

    def extract_highlight_clip(self, video_path: str, start: float, end: float, output_path: str) -> bool:
        """Extract a highlight clip with enhanced audio handling."""
        try:
            duration = end - start
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-b:a', '192k',     # Higher audio bitrate
                '-ar', '48000',     # Higher sample rate
                '-filter_complex',
                f'[0:v]fade=t=in:st=0:d=0.3,fade=t=out:st={duration-0.3}:d=0.3[v];' +
                f'[0:a]afade=t=in:st=0:d=0.3,afade=t=out:st={duration-0.3}:d=0.3[a]',
                '-map', '[v]',
                '-map', '[a]',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract clip: {str(e)}")
            return False

    def detect_movie_genres(self, dialogue_segments: List[Dict], audio_features: List[Dict]) -> Set[str]:
        """Enhanced genre detection with stricter thresholds."""
        genre_scores = defaultdict(float)
        
        # Analyze dialogue emotions with weighted scoring
        for segment in dialogue_segments:
            for genre, config in self.genre_config.items():
                # Emotion match with higher weight for strong emotions
                emotion_score = max(
                    segment["emotions"].get(emotion, 0)
                    for emotion in config["emotions"]
                )
                genre_scores[genre] += emotion_score * 1.5
                
                # Keyword analysis with context
                text = segment["text"].lower()
                keyword_matches = sum(1 for keyword in config["keywords"] if keyword in text)
                genre_scores[genre] += keyword_matches * 0.3
        
        # Analyze audio features with genre-specific criteria
        for feature in audio_features:
            # Action: high energy, fast tempo, strong onset
            if (feature["energy"] > 0.7 and 
                feature["tempo"] > 120 and 
                feature["onset_strength"] > 0.6):
                genre_scores["action"] += 0.8
            
            # Drama: moderate energy, varied spectral content, clear speech
            if (0.3 <= feature["energy"] <= 0.6 and
                feature["spectral_centroid"] > np.mean([f["spectral_centroid"] for f in audio_features])):
                genre_scores["drama"] += 0.5
            
            # Comedy: varied energy, high spectral variance
            if (feature["energy"] > 0.4 and
                feature["spectral_centroid"] > np.mean([f["spectral_centroid"] for f in audio_features]) * 1.2):
                genre_scores["comedy"] += 0.4
            
            # Romance: smooth audio features, lower energy
            if (feature["energy"] < 0.5 and
                feature["mfcc_mean"] > np.mean([f["mfcc_mean"] for f in audio_features])):
                genre_scores["romance"] += 0.4
        
        # Normalize scores
        total_segments = max(len(dialogue_segments), 1)
        for genre in genre_scores:
            genre_scores[genre] /= total_segments
        
        # Select genres that meet their specific thresholds
        selected_genres = {
            genre for genre, score in genre_scores.items()
            if score > self.genre_config[genre]["threshold"]
        }
        
        return selected_genres

    def generate_trailers(self, video_path: str) -> Dict[str, str]:
        """Generate multiple genre-specific trailers with improved quality control."""
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
            
            # Detect movie genres with stricter criteria
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

# Example usage
if __name__ == "__main__":
    generator = SmartTrailerGenerator(output_dir="output_trailers")
    video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"
    trailers = generator.generate_trailers(video_path)
    
    print("\nGenerated trailers:")
    for genre, path in trailers.items():
        print(f"{genre}: {path}")