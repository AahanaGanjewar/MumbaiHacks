import streamlit as st
import os
import time
from main7 import SmartTrailerGenerator

# Title of the app
st.title("AI-Powered Movie Trailer Generation")

# Meaningful description
st.markdown("""
## Generate Multiple Trailers from a Single Movie
One trailer per movie doesn't quite cover everyone's favorite parts.  
Can AI generate trailers with different highlights for varying demographic needs?  
Upload your movie, and we'll generate multiple trailers for you to choose from!
""")

# File uploader for video
uploaded_video = st.file_uploader("Upload your movie video here", type=["mp4", "avi", "mov"])

# Ensure the 'temp_videos' directory exists
temp_dir = "temp_videos"
os.makedirs(temp_dir, exist_ok=True)

# Placeholder for videos to be generated and displayed later
generated_videos = []

# Use your backend trailer generator
def generate_trailers(video_file_path):
    generator = SmartTrailerGenerator(output_dir="output_trailers")
    return generator.generate_trailers(video_file_path)

# Trigger video generation when upload button is clicked
if st.button("Generate Trailers"):
    if uploaded_video is not None:
        # Save the uploaded video temporarily
        video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.write("Processing your video... Please wait.")
        # Generate video trailers using your backend
        generated_videos = generate_trailers(video_path)
        st.success("Trailers generated successfully!")

        # Display generated videos
        st.markdown("### Watch your generated trailers:")
        for genre, path in generated_videos.items():
            st.video(path)  # Replace with correct path or URL to display the video
    else:
        st.error("Please upload a video first.")
