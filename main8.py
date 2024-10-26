import os
import torch
import ffmpeg
import numpy as np
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    pipeline
)
from moviepy.editor import *
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
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.genre_config = {
            "action": {
                "keywords": ["fight", "battle", "explosion", "chase", "gun", "run", "danger", "attack", "war", "fast"],
                "emotions": ["anger", "fear", "surprise"],
                "visual_cues": ["action scene", "explosion", "fight scene", "chase scene", "combat", "stunt"],
                "audio_features": {
                    "energy_threshold": 0.6,
                    "tempo_min": 110
                },
                "cinematic_effects": {
                    "color_grade": "high_contrast",
                    "speed": "dynamic",
                    "transition": "fast_cut",
                    "audio_effects": ["impact_sound", "bass_boost"],
                    "score_type": "intense"
                },
                "threshold": 0.55
            },
            "drama": {
                "keywords": ["emotional", "conflict", "relationship", "struggle", "family", "crisis", "truth", "secret"],
                "emotions": ["sadness", "neutral", "anger", "disgust"],
                "visual_cues": ["emotional scene", "dramatic moment", "intense dialogue", "confrontation"],
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
                "threshold": 0.6
            },
            "comedy": {
                "keywords": ["funny", "laugh", "joke", "humor", "smile", "silly", "fun", "crazy", "ridiculous", "hilarious"],
                "emotions": ["joy", "surprise", "neutral"],
                "visual_cues": ["funny moment", "comedic scene", "laughter", "silly action", "gag", "prank"],
                "audio_features": {
                    "laughter_threshold": 0.4,
                    "speech_rate_variance": 0.3,
                    "energy_variance": 0.4
                },
                "cinematic_effects": {
                    "color_grade": "vibrant",
                    "speed": "varied",
                    "transition": "whip_pan",
                    "audio_effects": ["quirky_sound", "upbeat"],
                    "score_type": "light"
                },
                "threshold": 0.5
            },
            "horror": {
                "keywords": ["scared", "fear", "terror", "dark", "scream", "monster", "ghost", "blood", "death", "nightmare"],
                "emotions": ["fear", "surprise", "disgust"],
                "visual_cues": ["dark scene", "scary moment", "horror element", "jumpscare", "creepy setting", "monster reveal"],
                "audio_features": {
                    "energy_threshold": 0.5,
                    "low_frequency_presence": 0.6,
                    "sudden_change_threshold": 0.7
                },
                "cinematic_effects": {
                    "color_grade": "dark",
                    "speed": "varied",
                    "transition": "flash",
                    "audio_effects": ["horror_ambient", "sudden_impact"],
                    "score_type": "suspense"
                },
                "threshold": 0.5
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
                "threshold": 0.6
            }
        }

    def apply_cinematic_effects(self, clip_path: str, genre: str, output_path: str) -> bool:
        """Apply genre-specific cinematic effects to video clip with improved error handling."""
        try:
            if genre not in self.genre_config:
                raise ValueError(f"Unknown genre: {genre}")

            # Get effects configuration
            genre_config = self.genre_config[genre]
            effects = genre_config.get("cinematic_effects", {
                "color_grade": "normal",
                "speed": "normal",
                "transition": "cut",
                "audio_effects": ["normalize"],
                "score_type": "standard"
            })
            
            # Build video filter
            video_filters = []
            
            # Color grading
            color_grade = effects.get("color_grade", "normal")
            if color_grade == "high_contrast":
                video_filters.append("eq=contrast=1.3:saturation=1.2:brightness=0.1")
            elif color_grade == "muted":
                video_filters.append("eq=contrast=0.9:saturation=0.8")
            elif color_grade == "vibrant":
                video_filters.append("eq=contrast=1.1:saturation=1.3")
            elif color_grade == "warm":
                video_filters.append("eq=contrast=1.1:saturation=1.1:gamma_r=1.1:gamma_b=0.9")
            elif color_grade == "dark":
                video_filters.append("eq=contrast=1.2:brightness=-0.1:saturation=0.8")
            
            # Speed effects
            speed = effects.get("speed", "normal")
            if speed == "dynamic":
                video_filters.append("setpts=0.8*PTS")
            elif speed == "slow":
                video_filters.append("setpts=1.2*PTS")
            elif speed == "varied":
                video_filters.append("setpts=if(lt(random(1),0.5),0.9*PTS,1.1*PTS)")
            
            # Build audio filter
            audio_filters = []
            audio_effects = effects.get("audio_effects", ["normalize"])
            
            if "impact_sound" in audio_effects:
                audio_filters.append("acompressor=threshold=0.125:ratio=20:attack=5:release=50")
                audio_filters.append("bass=g=3:frequency=100:width_type=h")
            elif "reverb" in audio_effects:
                audio_filters.append("aecho=0.8:0.8:40|50|70:0.4|0.3|0.2")
            elif "horror_ambient" in audio_effects:
                audio_filters.append("acompressor=threshold=0.1:ratio=3")
                audio_filters.append("lowpass=f=200,highpass=f=20")
            elif "soft_piano" in audio_effects:
                audio_filters.append("lowpass=f=3000,highpass=f=200")
                audio_filters.append("acompressor=threshold=0.1:ratio=2:attack=5:release=50")
            elif "quirky_sound" in audio_effects:
                audio_filters.append("vibrato=f=4:d=0.5")
                audio_filters.append("acompressor=threshold=0.2:ratio=2")
            else:
                audio_filters.append("dynaudnorm")
            
            # Combine filters into filter complex
            filter_complex = []
            
            # Add video filter chain
            if video_filters:
                video_filter_str = ','.join(video_filters)
                filter_complex.append(f"[0:v]{video_filter_str}[v]")
            else:
                filter_complex.append("[0:v]copy[v]")
            
            # Add audio filter chain
            if audio_filters:
                audio_filter_str = ','.join(audio_filters)
                filter_complex.append(f"[0:a]{audio_filter_str}[a]")
            else:
                filter_complex.append("[0:a]anull[a]")
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', clip_path,
                '-filter_complex',
                ';'.join(filter_complex),
                '-map', '[v]',
                '-map', '[a]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]
            
            # Run ffmpeg command
            subprocess.run(cmd, check=True, capture_output=True)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply cinematic effects: {str(e)}")
            return False
    # Rest of the class implementation remains the same...
    
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

    def calculate_genre_score(self, genre: str, visual_score: float, dialogue_segments: List[Dict], audio_features: List[Dict]) -> float:
        """Enhanced genre score calculation with better weighting."""
        genre_config = self.genre_config[genre]
        score = 0
        
        # Visual score weight (30%)
        score += visual_score * 0.3
        
        # Emotion score (25%)
        if dialogue_segments:
            emotion_scores = []
            for segment in dialogue_segments:
                # Calculate emotion score with weighted relevance
                relevant_emotions = []
                for emotion in genre_config["emotions"]:
                    emotion_value = segment["emotions"].get(emotion, 0)
                    if emotion == "fear" and genre == "horror":
                        emotion_value *= 1.5  # Boost fear detection for horror
                    elif emotion == "joy" and genre == "comedy":
                        emotion_value *= 1.3  # Boost joy detection for comedy
                    relevant_emotions.append(emotion_value)
                emotion_scores.append(max(relevant_emotions))
            score += (sum(emotion_scores) / len(emotion_scores)) * 0.25
        
        # Audio feature score (25%)
        if audio_features:
            audio_score = 0
            for feature in audio_features:
                if genre == "action":
                    audio_score = max(audio_score, 
                                    feature["energy"] * 0.6 + 
                                    (feature["tempo"] / 180) * 0.4)
                elif genre == "horror":
                    sudden_change = abs(feature["energy"] - feature.get("prev_energy", feature["energy"]))
                    audio_score = max(audio_score,
                                    feature["energy"] * 0.4 +
                                    feature.get("low_frequency_presence", 0) * 0.3 +
                                    sudden_change * 0.3)
                elif genre == "comedy":
                    energy_variance = abs(feature["energy"] - feature.get("prev_energy", feature["energy"]))
                    audio_score = max(audio_score,
                                    energy_variance * 0.5 +
                                    feature.get("speech_rate_variance", 0) * 0.5)
                # Other genres remain the same...
            
            score += audio_score * 0.25
        
        # Keyword presence score (20%)
        if dialogue_segments:
            keyword_score = 0
            total_words = 0
            for segment in dialogue_segments:
                words = segment["text"].lower().split()
                total_words += len(words)
                keyword_matches = sum(1 for word in words if word in genre_config["keywords"])
                if keyword_matches > 0:
                    keyword_score += keyword_matches / len(words)
            
            if total_words > 0:
                score += (keyword_score / len(dialogue_segments)) * 0.2
        
        return score

    def detect_movie_genres(self, dialogue_segments: List[Dict], audio_features: List[Dict]) -> Set[str]:
        """Enhanced genre detection with better handling of multiple genres."""
        genre_scores = defaultdict(float)
        
        # Calculate initial scores
        for segment in dialogue_segments:
            for genre, config in self.genre_config.items():
                # Emotion scoring with genre-specific weighting
                emotion_score = 0
                for emotion in config["emotions"]:
                    value = segment["emotions"].get(emotion, 0)
                    if genre == "horror" and emotion == "fear":
                        value *= 1.5
                    elif genre == "comedy" and emotion == "joy":
                        value *= 1.3
                    emotion_score = max(emotion_score, value)
                
                genre_scores[genre] += emotion_score * 1.2
                
                # Enhanced keyword analysis
                text = segment["text"].lower()
                words = set(text.split())
                keyword_matches = sum(1 for keyword in config["keywords"] if keyword in words)
                genre_scores[genre] += keyword_matches * 0.4
        
        # Audio feature analysis with genre-specific patterns
        prev_energy = None
        for feature in audio_features:
            current_energy = feature["energy"]
            
            # Action detection
            if feature["energy"] > 0.6 and feature["tempo"] > 110:
                genre_scores["action"] += 0.8
            
            # Horror detection (sudden changes and low frequencies)
            if prev_energy is not None:
                energy_change = abs(current_energy - prev_energy)
                if energy_change > 0.3:  # Sudden change
                    genre_scores["horror"] += 0.6
                if feature.get("low_frequency_presence", 0) > 0.5:
                    genre_scores["horror"] += 0.4
            
            # Comedy detection (varied energy patterns)
            if prev_energy is not None:
                energy_variance = abs(current_energy - prev_energy)
                if 0.2 < energy_variance < 0.6:  # Characteristic of comedy timing
                    genre_scores["comedy"] += 0.5
            
            prev_energy = current_energy
        
        # Normalize scores
        total_segments = max(len(dialogue_segments), 1)
        max_score = max(genre_scores.values()) if genre_scores else 1
        
        normalized_scores = {
            genre: (score / total_segments) / max_score
            for genre, score in genre_scores.items()
        }
        
        # Select genres with more lenient thresholds
        selected_genres = {
            genre for genre, score in normalized_scores.items()
            if score > self.genre_config[genre]["threshold"]
        }
        
        return selected_genres


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
