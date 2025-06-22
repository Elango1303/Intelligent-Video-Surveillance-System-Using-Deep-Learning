# Video Crime Detection System

A Flask-based web application that detects criminal activities in videos using a 3D Convolutional Neural Network (C3D) model. The system performs real-time analysis and can send WhatsApp alerts when specific criminal activities are detected.

## 🔍 Features

- **Real-time Crime Detection**: Identifies 14 types of criminal activities including:
  - Abuse, Arrest, Arson, Assault, Burglary
  - Explosion, Fighting, Robbery, Shooting
  - Shoplifting, Stealing, Vandalism

- **Instant Alerts**: 
  - Automated WhatsApp notifications to authorized personnel
  - Configurable alert triggers for specific crime types

- **Detailed Analysis**:
  - Frame-by-frame prediction visualization
  - Confidence score for each detection
  - Processing time metrics
  - Prediction distribution across video segments

## 🛠️ Technical Specifications

- **Core Model**: 3D Convolutional Neural Network (C3D)
- **Input**: Video clips (MP4, AVI, MOV, MKV)
- **Output**: Crime classification with confidence score
- **Performance**: Optimized for both CPU and GPU inference

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/video-crime-detection.git
   cd video-crime-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model weights** and place in `outputs/best_model.pth`

## 🚀 Usage

1. **Start the web application**:
   ```bash
   python app.py
   ```

2. **Access the web interface** at `http://localhost:5000`

3. **Upload a video file** and configure alert settings

4. **View results** including:
   - Original and annotated videos
   - Prediction statistics
   - Alert status

## ⚙️ Configuration

Edit `app.py` to configure:

```python
# Twilio configuration for WhatsApp alerts
app.config['TWILIO_ACCOUNT_SID'] = 'your_account_sid'
app.config['TWILIO_AUTH_TOKEN'] = 'your_auth_token'
app.config['TWILIO_FROM_NUMBER'] = 'whatsapp:+your_twilio_number'
app.config['FIXED_RECEIVER_NUMBER'] = '+911234567890'  # Fixed recipient number

# Model and file paths
app.config['MODEL_PATH'] = 'path_to_model_weights.pth'
app.config['UPLOAD_FOLDER'] = 'path_to_upload_directory'
```

## 📊 API Endpoints

- `POST /api/predict` - Analyze video programmatically
- `GET /` - Main upload interface
- `POST /upload` - Handle video uploads
- `GET /process/<filename>` - Process uploaded video

**Sample API request**:
```bash
curl -X POST -F "video=@test.mp4" http://localhost:5000/api/predict
```

## 📋 Dataset

The model was trained on the UCF Crime Dataset containing:
- 1,266,345 training images across 14 categories
- 111,308 test images

**Dataset structure**:
```
datasets/
├── Train/
│   ├── Abuse/
│   ├── Arrest/
│   └── ...other categories
└── Test/
    ├── Abuse/
    ├── Arrest/
    └── ...other categories
```

## 🧠 Model Architecture

The C3D model consists of:

```python
5x 3D convolutional layers
3x fully connected layers
Dropout for regularization
Batch normalization
```

## 📈 Performance Metrics

- **Training Accuracy**: 94.2%
- **Validation Accuracy**: 89.7%
- **Inference Time**: ~2.5 seconds per 16-frame clip (on NVIDIA Tesla T4)

