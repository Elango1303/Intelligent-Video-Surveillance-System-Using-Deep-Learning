import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# Define dataset paths
train_dir = 'D:/project_2/anomaly_detection/datasets/Train'
test_dir = 'D:/project_2/anomaly_detection/datasets/Test'

class UCFCrimeDataset(Dataset):
    """UCF Crime dataset for video anomaly detection
    
    The dataset contains images extracted from every video from the UCF Crime Dataset.
    Every 10th frame is extracted from each full-length video for each class.
    All images are of size 64x64 and in PNG format.
    
    Total train images: 1,266,345
    Total test images: 111,308
    """
    
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        
        self.classes = {
            'Abuse': 0, 'Arrest': 1, 'Arson': 2, 'Assault': 3, 
            'Burglary': 4, 'Explosion': 5, 'Fighting': 6, 'Normal': 7,
            'RoadAccidents': 8, 'Robbery': 9, 'Shooting': 10, 
            'Shoplifting': 11, 'Stealing': 12, 'Vandalism': 13
        }
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes.keys():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.png')]
                self.image_paths.extend(class_images)
                self.labels.extend([self.classes[class_name]] * len(class_images))
        
        # Create a mapping for extracting clips
        self.clips = []
        
        # Group images by class
        class_images = {}
        for path, label in zip(self.image_paths, self.labels):
            class_name = list(self.classes.keys())[label]
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(path)
        
        # Create clips for each class
        for class_name, paths in class_images.items():
            paths.sort()  # Sort to ensure sequential order
            
            # Create overlapping clips with stride of clip_len//2
            stride = max(1, clip_len // 2)
            for i in range(0, len(paths) - clip_len + 1, stride):
                clip_paths = paths[i:i + clip_len]
                if len(clip_paths) == clip_len:  # Ensure we have enough frames
                    self.clips.append((clip_paths, self.classes[class_name]))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        print(f"Created {len(self.clips)} clips for training/validation")
        
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        frame_paths, label = self.clips[idx]
        
        # Load frames
        clip = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                frame = self.transform(frame)
            
            clip.append(frame)
        
        # Stack frames into a clip tensor
        if isinstance(clip[0], torch.Tensor):
            # If transform already converts to tensor
            clip = torch.stack(clip, dim=0)  # [clip_len, C, H, W]
            # Rearrange to [C, clip_len, H, W] format for 3D CNN
            clip = clip.permute(1, 0, 2, 3)
        else:
            # If transform doesn't convert to tensor
            clip = np.array(clip).transpose((3, 0, 1, 2))  # [C, clip_len, H, W]
            clip = torch.from_numpy(clip).float()
        
        return clip, label

def extract_frames(video_path, output_dir, frame_interval=10, resize_shape=(64, 64)):
    """Extract frames from a video file at specified intervals and resize them
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 10)
        resize_shape: Tuple of (width, height) to resize frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Resize frame
            resized_frame = cv2.resize(frame, resize_shape)
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(frame_path, resized_frame)
            saved_count += 1
        
        frame_count += 1
    
    video.release()
    print(f"Extracted {saved_count} frames from {video_path}")

def analyze_dataset(dataset_dir):
    """Analyze the dataset structure and print statistics"""
    print(f"\nAnalyzing dataset in {dataset_dir}...")
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Directory not found: {dataset_dir}")
        return
    
    total_images = 0
    classes = {}
    
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        class_images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        num_images = len(class_images)
        
        classes[class_name] = num_images
        total_images += num_images
    
    print(f"Found {len(classes)} classes with {total_images} total images")
    print("\nClass distribution:")
    for class_name, count in sorted(classes.items()):
        print(f"  - {class_name}: {count} images ({count/total_images:.2%})")
    
    print(f"\nTotal image count: {total_images}")

def main():
    parser = argparse.ArgumentParser(description="Process UCF Crime dataset")
    parser.add_argument('--analyze', action='store_true', help='Analyze existing dataset structure')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("\n=== Analyzing Train Dataset ===")
        analyze_dataset(train_dir)
        
        print("\n=== Analyzing Test Dataset ===")
        analyze_dataset(test_dir)
    else:
        # By default, just print the information
        print("\nUCF Crime Dataset Information:")
        print("------------------------------")
        print("Dataset contains images extracted from UCF Crime Dataset videos")
        print("Every 10th frame extracted from each full-length video")
        print("All images are 64x64 PNG format")
        print("")
        print("14 Classes:")
        for i, class_name in enumerate([
            'Abuse', 'Arrest', 'Arson', 'Assault', 
            'Burglary', 'Explosion', 'Fighting', 'Normal',
            'RoadAccidents', 'Robbery', 'Shooting', 
            'Shoplifting', 'Stealing', 'Vandalism'
        ]):
            print(f"{i+1}. {class_name}")
        print("")
        print("Train subset: 1,266,345 images")
        print("Test subset: 111,308 images")
        print("")
        print("To analyze the current dataset structure, run with --analyze flag")

if __name__ == "__main__":
    main()