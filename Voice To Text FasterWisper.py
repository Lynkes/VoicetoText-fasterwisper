from faster_whisper import WhisperModel
import os
import tkinter as tk
from tkinter import filedialog

model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")


def save_segments(input_file_path, segments):
    # Get the base filename from the input path
    filename, _ = os.path.splitext(os.path.basename(input_file_path))
    
    # Create the output filename with .txt extension
    output_file_path = filename + '.txt'
    
    # Write segments to the output file
    with open(output_file_path, 'w') as output_file:
        for segment in segments:
            output_file.write(segment + '\n')


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory(title="Select Folder")
    
    if folder_path:
        print(f"Selected Folder: {folder_path}")
        return folder_path
    else:
        print("No folder selected.")
            

def process_mp4_files(folder_path):
    # List all files in the specified folder
    files = os.listdir(folder_path)
    
    # Filter for .mp4 files
    mp4_files = [file for file in files if file.lower().endswith('.mp4')]
    
    # Process each .mp4 file
    for mp4_file in mp4_files:
        # Construct the full path to the .mp4 file
        mp4_file_path = os.path.join(folder_path, mp4_file)
        
        # Process the .mp4 file (replace this with your own processing logic)
        print(f"Processing file: {mp4_file_path}")
        
        # Example: You can add your segmentation or other processing logic here
        # For instance, if you wanted to process segments:
        segments, info = model.transcribe(mp4_file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        # Get the base filename from the input path
        filename, _ = os.path.splitext(os.path.basename(mp4_file_path))
        # Create the output filename with .txt extension
        output_file_path = folder_path + filename + '.txt'

        # Write segments to the output file
        with open(output_file_path, 'w') as output_file:
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                output_file.write(segment.text + '\n')


process_mp4_files(select_folder())
