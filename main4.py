import os
import torch
import ffmpeg
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import numpy as np
import re

from main import classify_video_genre, load_clip_model

# Step 1: Extract Audio from Video
def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)

# Step 2: Transcribe Audio
def transcribe_audio(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    audio_input = AudioSegment.from_file(audio_path)
    audio_input = audio_input.set_frame_rate(16000)  # Set to 16kHz
    audio_samples = np.array(audio_input.get_array_of_samples(), dtype=np.float32)  # Ensure float32

    # Normalize audio samples to range [-1, 1]
    audio_samples /= np.max(np.abs(audio_samples))

    # Use the Wav2Vec2 model for transcription
    input_values = processor(audio_samples, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]  # Assuming single transcription
# Step 3: Analyze Transcription for Genres
def analyze_transcription(transcription, genres):
    # Basic keyword matching for genre classification
    genre_keywords = {
        "action": ["fight", "battle", "explosion"],
        "comedy": ["funny", "joke", "laugh"],
        "drama": ["cry", "love", "relationship"],
        "horror": ["scare", "ghost", "fear"],
        "sci-fi": ["alien", "space", "future"],
        "romance": ["love", "date", "kiss"],
        "thriller": ["mystery", "suspense", "crime"]
    }

    genre_count = {genre: 0 for genre in genres}
    
    for genre, keywords in genre_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', transcription, re.IGNORECASE):
                genre_count[genre] += 1

    predicted_genre = max(genre_count, key=genre_count.get)
    return predicted_genre

# Step 4: Extract Clips Based on Dialogue
def extract_clips_based_on_dialogue(video_path, model, processor, genres):
    audio_path = "audio.wav"
    extract_audio(video_path, audio_path)
    transcription = transcribe_audio(audio_path)

    # Classify genre based on transcription
    predicted_genre = analyze_transcription(transcription, genres)
    
    # Use the start and end times of dialogues (if known) for extraction
    # This is a placeholder; in practice, you would need to identify dialogue timestamps
    start_time = 0  # Adjust as needed
    end_time = 10  # Adjust as needed

    # Create temporary video clip for the dialogue
    clip_output_path = f"dialogue_clip_{predicted_genre}.mp4"
    (
        ffmpeg
        .input(video_path, ss=start_time, to=end_time)  # Adjust timing logic as needed
        .output(clip_output_path)
        .run(overwrite_output=True)
    )
    
    return {predicted_genre: clip_output_path}

# Step 5: Combine Clips by Genre
def combine_clips(genre_clips, genre):
    if genre in genre_clips:
        clip_list = genre_clips[genre]
        with open(r'C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\temp_file.txt', 'w') as f:
            f.write(f"file '{clip_list}'\n")
        
        # Combine clips using ffmpeg
        combined_output = f"trailer_{genre}.mp4"
        ffmpeg.input('temp_file.txt', format='concat', safe=0).output(combined_output).run()
        os.remove('temp_file.txt')  # Clean up

# Step 6: Main Function
def create_trailers(video_path):
    # Load the CLIP model
    model, processor = load_clip_model()

    # Define possible genres
    genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller"]

    # Extract clips based on dialogue
    genre_clips = extract_clips_based_on_dialogue(video_path, model, processor, genres)

    # Combine clips for each genre
    for genre in genres:
        combine_clips(genre_clips, genre)

# Usage Example
video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"
create_trailers(video_path)