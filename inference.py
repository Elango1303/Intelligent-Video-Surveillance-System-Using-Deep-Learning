import os
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from models.cnn3d import C3D
from torchvision import transforms
from utils.video_utils import extract_frames_from_video, process_video_frames

def load_model(model_path, num_classes=14, device='cuda'):
    """Load a trained model"""
    model = C3D(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict_video(model, video_path, clip_len=16, device='cuda'):
    """Predict anomaly class for a video"""
    # Extract frames from video
    frames = extract_frames_from_video(video_path)
    
    # Process frames for prediction
    clips = process_video_frames(frames, clip_len=clip_len)
    
    # Transform for inference
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),  # Changed to 112x112 to match expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each clip
    predictions = []
    confidences = []
    
    for clip in tqdm(clips, desc="Processing clips"):
        # Apply transformation to each frame
        processed_clip = []
        for frame in clip:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            processed_clip.append(frame)
        
        # Stack frames along a new dimension to create clip tensor
        # This creates a tensor of shape [clip_len, channels, height, width]
        processed_clip = torch.stack(processed_clip)
        
        # Reorder dimensions to [channels, clip_len, height, width] as expected by C3D
        processed_clip = processed_clip.permute(1, 0, 2, 3)
        
        # Add batch dimension
        processed_clip = processed_clip.unsqueeze(0)
        
        # Debug: Print tensor shape
        print(f"Clip shape before model: {processed_clip.shape}")
        
        # Move to device
        processed_clip = processed_clip.to(device)
        
        # Get prediction
        try:
            with torch.no_grad():
                outputs = model(processed_clip)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = probs.max(1)
                
                predictions.append(prediction.item())
                confidences.append(confidence.item())
        except RuntimeError as e:
            print(f"Error during prediction: {e}")
            # We'll continue with other clips
            continue
    
    # Get the most common prediction and average confidence
    if predictions:
        pred_counts = np.bincount(predictions)
        most_common_pred = np.argmax(pred_counts)
        avg_confidence = np.mean([confidences[i] for i, p in enumerate(predictions) if p == most_common_pred])
        
        return most_common_pred, avg_confidence, predictions, confidences
    else:
        return None, 0, [], []

class ModelWrapper(torch.nn.Module):
    """Wrapper around C3D model to handle feature resizing if needed"""
    
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
        # Extract the fully connected layers
        self.fc6 = model.fc6
        self.fc7 = model.fc7
        self.fc8 = model.fc8
        self.dropout = model.dropout
        
        # Save expected input size to fc6
        self.fc6_in_features = self.fc6.in_features
        
    def forward(self, x):
        # Run through convolutional layers
        x = self.model.pool1(torch.relu(self.model.conv1(x)))
        x = self.model.pool2(torch.relu(self.model.conv2(x)))
        
        x = torch.relu(self.model.conv3a(x))
        x = torch.relu(self.model.conv3b(x))
        x = self.model.pool3(x)
        
        x = torch.relu(self.model.conv4a(x))
        x = torch.relu(self.model.conv4b(x))
        x = self.model.pool4(x)
        
        x = torch.relu(self.model.conv5a(x))
        x = torch.relu(self.model.conv5b(x))
        x = self.model.pool5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Debug
        print(f"Feature shape before FC layers: {x.shape}")
        
        # Handle shape mismatch
        if x.shape[1] != self.fc6_in_features:
            print(f"Warning: Feature size mismatch. Got {x.shape[1]}, expected {self.fc6_in_features}. Adjusting...")
            # Option 1: Interpolate to the right size
            x = x.view(x.size(0), -1)
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1).unsqueeze(1), 
                size=(1, self.fc6_in_features),
                mode='bilinear'
            ).squeeze(1).squeeze(1)
        
        # Continue with FC layers
        x = torch.relu(self.fc6(x))
        x = self.dropout(x)
        
        x = torch.relu(self.fc7(x))
        x = self.dropout(x)
        
        x = self.fc8(x)
        
        return x

def main():
    parser = argparse.ArgumentParser(description="Inference on video for anomaly detection")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--clip_len', type=int, default=16, help='Number of frames in a clip')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    base_model = load_model(args.model_path, device=device)
    # Wrap the model to handle feature resizing
    model = ModelWrapper(base_model)
    model = model.to(device)
    model.eval()
    
    # Define class names
    class_names = [
        'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
        'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 
        'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
    ]
    
    # Predict video
    print(f"Processing video: {args.video_path}")
    pred_class, confidence, all_preds, all_confs = predict_video(
        model, args.video_path, args.clip_len, device
    )
    
    if pred_class is not None:
        print(f"Predicted class: {class_names[pred_class]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Count predictions for each class
        print("\nPrediction distribution across clips:")
        for i, count in enumerate(np.bincount(all_preds, minlength=len(class_names))):
            if count > 0:
                print(f"{class_names[i]}: {count} clips ({count/len(all_preds)*100:.1f}%)")
    else:
        print("No predictions could be made for this video.")

if __name__ == "__main__":
    main()