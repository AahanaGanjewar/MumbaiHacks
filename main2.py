import cv2
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import ffmpeg
import librosa

# Step 1: Extract Visual Frames from the Video
def extract_frames(video_path, output_folder, num_frames=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    frame_count = 0
    extracted = 0
    while video_cap.isOpened() and extracted < num_frames:
        ret, frame = video_cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_path = os.path.join(output_folder, f"frame_{extracted}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted += 1
        frame_count += 1

    video_cap.release()

# Step 2: Extract Audio from the Video
def extract_audio(video_path, audio_output):
    ffmpeg.input(video_path).output(audio_output).run(cmd=r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe", overwrite_output=True)

# Load Models for Visual and Audio
def load_models():
    visual_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    visual_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\wav2vec2-base-superb-ks")
    audio_processor = Wav2Vec2Processor.from_pretrained(r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\wav2vec2-base-superb-ks")

    return visual_model, visual_processor, audio_model, audio_processor

# Classify Visual Content
def classify_visual(frames_folder, visual_model, visual_processor, genres):
    images = []
    for frame_file in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_file)
        image = Image.open(frame_path).convert("RGB")
        images.append(image)

    inputs = visual_processor(text=genres, images=images, return_tensors="pt", padding=True)
    outputs = visual_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    avg_probs = torch.mean(probs, dim=0)
    genre_idx = torch.argmax(avg_probs).item()
    return genres[genre_idx]

# Classify Audio Content
def classify_audio(audio_path, audio_model, audio_processor):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = audio_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = audio_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    return predicted_ids.item()

# Combine Predictions
def combine_predictions(visual_genre, audio_genre):
    votes = [visual_genre, audio_genre]
    final_genre = max(set(votes), key=votes.count)
    return final_genre

# Main Function
def classify_video_genre(video_path):
    frames_folder = "extracted_frames"
    audio_output = "audio.wav"

    extract_frames(video_path, frames_folder, num_frames=10)
    extract_audio(video_path, audio_output)

    visual_model, visual_processor, audio_model, audio_processor = load_models()
    genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller"]

    visual_genre = classify_visual(frames_folder, visual_model, visual_processor, genres)
    audio_genre = classify_audio(audio_output, audio_model, audio_processor)

    final_genre = combine_predictions(visual_genre, audio_genre)

    # Cleanup
    if os.path.exists(frames_folder):
        for file in os.listdir(frames_folder):
            os.remove(os.path.join(frames_folder, file))
        os.rmdir(frames_folder)
    if os.path.exists(audio_output):
        os.remove(audio_output)

    return final_genre

# Video path
video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"
predicted_genre = classify_video_genre(video_path)
print(f"The predicted genre of the video is: {predicted_genre}")
