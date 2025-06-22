import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import glob
from collections import OrderedDict
from torch.cuda.amp import GradScaler, autocast  # Import for mixed precision

# Define the C3D model
class C3D(nn.Module):
    def __init__(self, num_classes=14):
        super(C3D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Fully connected layers
        self.fc6 = nn.Linear(8192, 4096)  # Fixed dimension here
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        
        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Adjust input shape if needed (B, C, T, H, W)
        if x.size(2) == 3:  # If channels are in the wrong dimension
            x = x.permute(0, 2, 1, 3, 4)
        
        # Convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        
        x = self.fc8(x)
        
        return x

# Dataset class
class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        
        self.classes = OrderedDict()
        class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for i, class_name in enumerate(class_dirs):
            self.classes[class_name] = i
        
        self.clips = []
        self._find_clips()
    
    def _find_clips(self):
        total_images = 0
        print(f"Finding clips in {self.root_dir}...")
        
        for class_name, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, class_name)
            images = sorted(glob.glob(os.path.join(class_dir, '*.png')))
            total_images += len(images)
            
            # Skip if no images found
            if len(images) == 0:
                continue
            
            # Group images into clips
            for i in range(0, len(images) - self.clip_len + 1, self.clip_len // 2):  # Overlap by half
                clip_paths = images[i:i + self.clip_len]
                if len(clip_paths) == self.clip_len:  # Ensure we have enough frames
                    self.clips.append((clip_paths, class_idx))
        
        print(f"Found {total_images} images in {self.root_dir}")
        print(f"Created {len(self.clips)} clips for training/validation")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip_paths, label = self.clips[idx]
        clip = []
        
        for frame_path in clip_paths:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            if self.transform:
                frame = self.transform(frame)
            
            clip.append(frame)
        
        # Stack frames to create a clip [C, T, H, W]
        clip = torch.stack(clip, dim=0).transpose(0, 1)
        
        return clip, label

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for clips, labels in progress_bar:
        clips = clips.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        with autocast():  # Mixed precision
            outputs = model(clips)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()  # Scale the loss for mixed precision
        scaler.step(optimizer)          # Update the optimizer
        scaler.update()                 # Update the scale for next iteration

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, train_dataset):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for clips, labels in progress_bar:
            clips = clips.to(device)
            labels = labels.to(device)
            
            with autocast():  # Mixed precision
                outputs = model(clips)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Save predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
    
    # Calculate metrics
    class_names = list(train_dataset.classes.keys())
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    
    return running_loss / len(dataloader), 100. * correct / total, conf_matrix, class_report

def verify_dataset_structure(train_dir, test_dir):
    """Verify dataset structure and print helpful information"""
    print(f"\n=== Checking dataset structure ===")
    
    # Check if train directory exists
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory {train_dir} does not exist!")
        return False
    
    # Check for class directories in Train
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not train_classes:
        print(f"ERROR: No class directories found in {train_dir}!")
        print("Expected class directories like: Abuse, Arrest, Arson, etc.")
        return False
    
    print(f"Found Train directory with {len(train_classes)} classes: {', '.join(train_classes)}")
    
    # Check for image files in each class
    total_train_images = 0
    for class_name in train_classes:
        class_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        if not images:
            print(f"WARNING: No PNG images found in {class_dir}!")
        else:
            print(f"  - {class_name}: {len(images)} images")
            total_train_images += len(images)
    
    print(f"Total train images: {total_train_images} (expected: 1,266,345)")
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"WARNING: Test directory {test_dir} does not exist!")
        return False
    else:
        test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if not test_classes:
            print(f"ERROR: No class directories found in {test_dir}!")
            return False
        
        print(f"Found Test directory with {len(test_classes)} classes.")
        
        # Check for image files in each test class
        total_test_images = 0
        for class_name in test_classes:
            class_dir = os.path.join(test_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            if not images:
                print(f"WARNING: No PNG images found in {class_dir}!")
            else:
                print(f"  - {class_name}: {len(images)} images")
                total_test_images += len(images)
        
        print(f"Total test images: {total_test_images} (expected: 111,308)")
    
    print("=== Dataset structure check complete ===\n")
    return True

def main(args):
    set_seed(args.seed)
    
    # Define paths
    train_dir = 'D:/project_2/anomaly_detection/datasets/Train'
    test_dir = 'D:/project_2/anomaly_detection/datasets/Test'
    
    # First, verify the dataset structure
    if not verify_dataset_structure(train_dir, test_dir):
        print("Dataset structure verification failed. Please fix the issues and try again.")
        return
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),  # Resize to standard C3D input
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),  # Resize to standard C3D input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print(f"Loading training data from: {train_dir}")
    train_dataset = UCFCrimeDataset(
        root_dir=train_dir,
        clip_len=args.clip_len,
        transform=train_transform
    )
    
    print(f"Loading test data from: {test_dir}")
    val_dataset = UCFCrimeDataset(
        root_dir=test_dir,
        clip_len=args.clip_len,
        transform=val_transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        print("ERROR: Train dataset is empty! Cannot continue training.")
        return
    
    if len(val_dataset) == 0:
        print("ERROR: Validation dataset is empty! Cannot continue training.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # Keep this smaller if you run into memory issues
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = C3D(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    # Print model summary
    print(model)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # If multi-GPU is available
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # TensorBoard writer
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, conf_matrix, class_report = validate(model, val_loader, criterion, device, train_dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("Classification Report:")
        print(class_report)
        
        # Save the model checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, checkpoint_path)
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_acc': best_acc,
            }, best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
    
    writer.close()
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CNN for anomaly detection on UCF Crime dataset")
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save models and logs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')  # Reduced batch size
    parser.add_argument('--clip_len', type=int, default=16, help='Number of frames in each clip')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')  # Changed to 5
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs if available')
    
    args = parser.parse_args()
    main(args)