import cv2
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import ffmpeg

# Step 1: Extract Frames from the Video
def extract_frames(video_path, output_folder, num_frames=10):
    """
    Extracts frames from a video.
    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save the extracted frames.
        num_frames (int): Number of frames to extract.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
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

# Step 2: Prepare the CLIP Model
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Step 3: Classify the Frames Using CLIP
def classify_video_frames(frames_folder, model, processor, genres):
    images = []
    for frame_file in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_file)
        image = Image.open(frame_path).convert("RGB")
        images.append(image)

    # Prepare the input
    inputs = processor(text=genres, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # Get probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Average the probabilities across all frames
    avg_probs = torch.mean(probs, dim=0)
    genre_idx = torch.argmax(avg_probs).item()
    return genres[genre_idx]

# Step 4: Main Function to Classify Video Genre
def classify_video_genre(video_path):
    # Extract frames
    output_folder = "extracted_frames"
    extract_frames(video_path, output_folder, num_frames=10)

    # Load the CLIP model
    model, processor = load_clip_model()

    # Define possible genres
    genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller"]

    # Classify the frames
    predicted_genre = classify_video_frames(output_folder, model, processor, genres)

    # Cleanup: Remove extracted frames
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))
    os.rmdir(output_folder)

    return predicted_genre

# Step 5: Usage Example
video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\3427514-uhd_3840_2160_24fps.mp4"
predicted_genre = classify_video_genre(video_path)
print(f"The predicted genre of the video is: {predicted_genre}")
