import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_frames_from_video(video_path, sample_rate=10):
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        sample_rate (int): Extract every Nth frame
        
    Returns:
        list: List of extracted frames as NumPy arrays
    """
    frames = []
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {frame_count} frames, {fps} FPS")
    
    # Extract frames
    for i in tqdm(range(frame_count), desc="Extracting frames"):
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the frame
        ret, frame = video.read()
        
        if ret:
            # Only keep every Nth frame
            if i % sample_rate == 0:
                frames.append(frame)
        else:
            break
    
    # Release the video
    video.release()
    
    print(f"Extracted {len(frames)} frames")
    return frames

def process_video_frames(frames, clip_len=16, overlap=0.5):
    """
    Process extracted frames into clips
    
    Args:
        frames (list): List of video frames
        clip_len (int): Number of frames in each clip
        overlap (float): Overlap between consecutive clips (0-1)
        
    Returns:
        list: List of clips (each clip is a list of frames)
    """
    if len(frames) < clip_len:
        # If we have fewer frames than clip_len, duplicate frames
        frames = frames * (clip_len // len(frames) + 1)
        frames = frames[:clip_len]
        return [frames]
    
    # Calculate step size with overlap
    step_size = max(1, int(clip_len * (1 - overlap)))
    
    # Create clips
    clips = []
    for i in range(0, len(frames) - clip_len + 1, step_size):
        clip = frames[i:i + clip_len]
        clips.append(clip)
    
    print(f"Created {len(clips)} clips from {len(frames)} frames")
    return clips

def save_prediction_visualization(video_path, predictions, confidences, class_names, output_path=None):
    """
    Create a visualization of the predictions overlaid on the video
    
    Args:
        video_path (str): Path to the original video
        predictions (list): List of class predictions for each clip
        confidences (list): List of confidence values for each prediction
        class_names (list): List of class names
        output_path (str): Path to save the output video
        
    Returns:
        str: Path to the output video
    """
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_prediction.mp4"
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate frames per clip
    frames_per_clip = frame_count / len(predictions)
    
    # Process each frame
    frame_idx = 0
    with tqdm(total=frame_count, desc="Creating visualization") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Calculate which clip this frame belongs to
            clip_idx = min(int(frame_idx / frames_per_clip), len(predictions) - 1)
            pred_class = predictions[clip_idx]
            confidence = confidences[clip_idx]
            
            # Add prediction text to frame
            text = f"{class_names[pred_class]}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Write the frame
            out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Release everything
    video.release()
    out.release()
    
    print(f"Visualization saved to {output_path}")
    return output_path