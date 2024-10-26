from main7 import SmartTrailerGenerator


generator = SmartTrailerGenerator(output_dir="output_trailers")
video_path = r"C:\Users\aahan\OneDrive\Desktop\MumbaiHacks\videoplayback(1).mp4"
trailers = generator.generate_trailers(video_path)
