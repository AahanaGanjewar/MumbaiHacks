import cv2
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import ffmpeg
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from main import classify_video_genre, load_clip_model

# Step 1: Scene Detection
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.start()
    
    # Detect scenes
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()

# Step 2: Extract and Classify Scenes
def extract_and_classify_scenes(video_path, model, processor, genres):
    scenes = detect_scenes(video_path)
    genre_clips = {genre: [] for genre in genres}
    
    for i, (start, end) in enumerate(scenes):
        start_frame = int(start.get_frames())
        end_frame = int(end.get_frames())
        
        # Create temporary video clip for the scene
        scene_output_path = f"scene_{i}.mp4"
        (
            ffmpeg
            .input(video_path, ss=start_frame / 24, to=end_frame / 24)  # Assuming 24 fps
            .output(scene_output_path)
            .run(overwrite_output=True)
        )
        
        # Classify the scene
        predicted_genre = classify_video_genre(scene_output_path)
        genre_clips[predicted_genre].append(scene_output_path)
        
    return genre_clips

# Step 3: Combine Clips by Genre
def combine_clips(genre_clips, genre):
    if genre in genre_clips:
        clip_list = genre_clips[genre]
        with open(r'C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\temp_file.txt', 'w') as f:
            for clip in clip_list:
                f.write(f"file '{clip}'\n")
        
        # Combine clips using ffmpeg
        combined_output = f"trailer_{genre}.mp4"
        ffmpeg.input('temp_file.txt', format='concat', safe=0).output(combined_output).run()
        os.remove('temp_file.txt')  # Clean up

# Step 4: Main Function
def create_trailers(video_path):
    # Load the CLIP model
    model, processor = load_clip_model()

    # Define possible genres
    genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller"]

    # Extract and classify scenes
    genre_clips = extract_and_classify_scenes(video_path, model, processor, genres)

    # Combine clips for each genre
    for genre in genres:
        combine_clips(genre_clips, genre)

# Usage Example
video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"
create_trailers(video_path)
