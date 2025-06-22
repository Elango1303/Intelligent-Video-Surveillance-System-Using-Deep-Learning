import os
import sys
import uuid
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import torch
from werkzeug.utils import secure_filename
from twilio.rest import Client  # Twilio import for WhatsApp integration

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.cnn3d import C3D
from inference import load_model, predict_video
from utils.video_utils import save_prediction_visualization

# Create template directory if it doesn't exist
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

os.makedirs(template_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "uploads"), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Create a basic index.html template if it doesn't exist
index_html_path = os.path.join(template_dir, "index.html")
if not os.path.exists(index_html_path):
    with open(index_html_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Video Crime Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-form {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            margin-top: 30px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        .message {
            margin: 20px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
        /* Alert section styles */
        .alert-settings {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .alert-settings h3 {
            margin-top: 0;
        }
        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        .phone-input {
            width: 100%;
            padding: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Video Crime Detection System</h1>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="message {{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    <div class="upload-form">
        <h2>Upload Video</h2>
        <p>Select a video file to analyze (MP4, AVI, MOV, MKV)</p>
        <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data">
            <input type="file" name="video" accept=".mp4,.avi,.mov,.mkv">
            
            <!-- WhatsApp Alert Settings - Always enabled with fixed number -->
            <div class="alert-settings">
                <h3>WhatsApp Alert Settings</h3>
                <p>Select which crime types should trigger alerts:</p>
                
                <div class="checkbox-group">
                    {% for class_name in class_names %}
                    <label>
                        <input type="checkbox" name="alert_classes" value="{{ class_name }}" 
                               {% if class_name in ["Shooting", "Robbery", "Explosion", "Assault", "Arson"] %}checked{% endif %}> 
                        {{ class_name }}
                    </label>
                    {% endfor %}
                </div>
                
                <p>WhatsApp alerts will be sent to: +91 7397555299</p>
                <input type="hidden" name="phone_number" value="+917397555299">
            </div>
            
            <br>
            <button type="submit" class="btn">Upload & Analyze</button>
        </form>
    </div>
</body>
</html>
""")

# Create a basic result.html template if it doesn't exist
result_html_path = os.path.join(template_dir, "result.html")
if not os.path.exists(result_html_path):
    with open(result_html_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Detection Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #333;
        }
        .result-container {
            margin-top: 30px;
        }
        .video-container {
            margin: 20px 0;
        }
        video {
            max-width: 100%;
        }
        .stats {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .alert-sent {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .alert-not-sent {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Detection Results</h1>
    
    <div class="stats">
        <h2>Prediction: {{ predicted_class }}</h2>
        <p><strong>Confidence:</strong> {{ "%.2f"|format(confidence*100) }}%</p>
        <p><strong>Processing Time:</strong> {{ "%.2f"|format(processing_time) }} seconds</p>
        
        <!-- Always show WhatsApp alert status -->
        {% if alert_sent %}
        <div class="alert-sent">
            <p><strong>WhatsApp Alert Sent!</strong> A notification was sent to {{ phone_number }}.</p>
        </div>
        {% else %}
        <div class="alert-not-sent">
            <p><strong>No Alert Sent:</strong> {{ alert_message }}</p>
        </div>
        {% endif %}
    </div>
    
    <div class="result-container">
        <h2>Original Video</h2>
        <div class="video-container">
            <video controls>
                <source src="{{ url_for('static', filename='uploads/' + original_video) }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <h2>Visualization with Predictions</h2>
        <div class="video-container">
            <video controls>
                <source src="{{ url_for('static', filename='uploads/' + prediction_video) }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <h2>Prediction Distribution</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {% for class_name, data in class_counts.items() %}
            <tr>
                <td>{{ class_name }}</td>
                <td>{{ data.count }}</td>
                <td>{{ "%.2f"|format(data.percentage) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <a href="{{ url_for('index') }}" class="btn">Analyze Another Video</a>
</body>
</html>
""")

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

app.config['SECRET_KEY'] = 'ucf_crime_detection_app'
app.config['UPLOAD_FOLDER'] = os.path.join(static_dir, 'uploads')
app.config['OUTPUT_FOLDER'] = 'D:/00001/output'
app.config['MODEL_PATH'] = 'D:/00001/outputs/best_model.pth'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Twilio configuration
# Twilio configuration
app.config['TWILIO_ACCOUNT_SID'] = os.environ.get('TWILIO_ACCOUNT_SID', 'ACxxxxxxxxxxxxxxxxxxxxx')
app.config['TWILIO_AUTH_TOKEN'] = os.environ.get('TWILIO_AUTH_TOKEN', 'xxxxxxxxxxxxxxxxxxxx')
app.config['TWILIO_FROM_NUMBER'] = os.environ.get('TWILIO_FROM_NUMBER', 'whatsapp:+xxxxx')  # <-- Twilio Sandbox Number
app.config['FIXED_RECEIVER_NUMBER'] = '+91xxxxxxxx'  # Fixed receiver number

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Class names
CLASS_NAMES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

# High severity classes that should trigger alerts by default
DEFAULT_ALERT_CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

# Load model on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Custom wrapper for torch.load to handle warnings
def load_model_safe(model_path, device):
    try:
        # First try with weights_only=True (PyTorch newer versions)
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # If that fails, fall back to the original method
        return torch.load(model_path, map_location=device)

# Initialize the model
def load_detection_model():
    global model
    try:
        print(f"Attempting to load model from {app.config['MODEL_PATH']}")
        
        # Use the custom wrapper for torch.load
        checkpoint = load_model_safe(app.config['MODEL_PATH'], device)
        
        # Initialize the model
        model = C3D(num_classes=len(CLASS_NAMES))
        
        # Check the structure of the checkpoint
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the checkpoint itself is the state dict
                model.load_state_dict(checkpoint)
        else:
            # If checkpoint is not a dictionary, it might be the model itself
            model = checkpoint
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        # Test the model with a small dummy input
        dummy_input = torch.zeros((1, 3, 16, 112, 112), device=device)
        with torch.no_grad():
            _ = model(dummy_input)
        print("Model successfully processed a test input")
        
        return model
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Generate a timestamped event-based filename
def generate_event_filename(original_filename, event_name, prefix=""):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    extension = original_filename.rsplit('.', 1)[1].lower()
    sanitized_event = event_name.replace(' ', '_').lower()
    return f"{prefix}{sanitized_event}_{timestamp}.{extension}"

# Function to send WhatsApp alert with Twilio
def send_whatsapp_alert(to_number, event_name, confidence, location="Unknown Location"):
    try:
        # Always use the fixed number regardless of input
        to_number = app.config['FIXED_RECEIVER_NUMBER'].strip('+')
        
        # Initialize Twilio client with your credentials
        client = Client(app.config['TWILIO_ACCOUNT_SID'], app.config['TWILIO_AUTH_TOKEN'])
        
        # Format the message
        message_body = f"⚠️ ALERT: {event_name} detected with {confidence:.2f}% confidence.\n"
        message_body += f"Location: {location}\n"
        message_body += f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message_body += "Please check the surveillance system immediately."
        
        # Send the WhatsApp message
        message = client.messages.create(
            body=message_body,
            from_=app.config['TWILIO_FROM_NUMBER'],
            to=f"whatsapp:+{to_number}"
        )
        
        print(f"WhatsApp alert sent to +{to_number}: {message.sid}")
        return True, message.sid
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)

@app.route('/')
def index():
    global model
    # Initialize the model if it's not loaded yet
    if model is None:
        model = load_detection_model()
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global model
    
    # Handle GET requests to /upload by redirecting to the index page
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    # Check if model is loaded
    if model is None:
        model = load_detection_model()  # Try to load model
        if model is None:  # If still None, show error
            flash('Model not loaded. Please try again later.', 'error')
            return redirect(url_for('index'))
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        flash('No video selected', 'error')
        return redirect(url_for('index'))
    
    # Get WhatsApp alert settings - alerts are always enabled with fixed number
    alert_classes = request.form.getlist('alert_classes')
    phone_number = app.config['FIXED_RECEIVER_NUMBER']  # Use fixed number
    
    # If no classes selected, use defaults
    if not alert_classes:
        alert_classes = DEFAULT_ALERT_CLASSES
    
    # Check if file type is allowed
    if file and allowed_file(file.filename):
        # Use temporary UUID for processing
        temp_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Save the file
        file.save(filepath)
        
        # Redirect to processing page with alert settings as query parameters
        return redirect(url_for('process_video', 
                               filename=temp_filename, 
                               original_filename=file.filename,
                               alert_classes=','.join(alert_classes),
                               phone_number=phone_number))
    
    flash('File type not allowed. Please upload mp4, avi, mov, or mkv.', 'error')
    return redirect(url_for('index'))

@app.route('/process/<filename>')
def process_video(filename):
    global model
    # Get original filename from query params if available
    original_filename = request.args.get('original_filename', filename)
    
    # Get alert settings from query parameters - use fixed number
    alert_classes = request.args.get('alert_classes', '').split(',') if request.args.get('alert_classes') else DEFAULT_ALERT_CLASSES
    phone_number = app.config['FIXED_RECEIVER_NUMBER']  # Always use fixed number
    
    # Ensure model is loaded
    if model is None:
        model = load_detection_model()
        if model is None:
            flash('Model not loaded. Please try again later.', 'error')
            return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        flash('File not found', 'error')
        return redirect(url_for('index'))
    
    try:
        # Process the video
        start_time = time.time()
        pred_class, confidence, all_preds, all_confs = predict_video(
            model, filepath, clip_len=16, device=device
        )
        processing_time = time.time() - start_time
        
        if pred_class is not None:
            # Create prediction distribution
            class_counts = {}
            for i, count in enumerate(torch.bincount(torch.tensor(all_preds), minlength=len(CLASS_NAMES))):
                if count > 0:
                    class_counts[CLASS_NAMES[i]] = {
                        'count': count.item(),
                        'percentage': count.item() / len(all_preds) * 100
                    }
            
            # Get the predicted event name
            event_name = CLASS_NAMES[pred_class]
            
            # Generate event-based filenames
            event_original_filename = generate_event_filename(original_filename, event_name, "original_")
            event_result_filename = generate_event_filename(original_filename, event_name, "result_")
            
            # Copy original file with event name
            original_event_path = os.path.join(app.config['OUTPUT_FOLDER'], event_original_filename)
            with open(filepath, 'rb') as src_file, open(original_event_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
            
            # Also keep a copy in uploads folder for web display
            display_original = os.path.join(app.config['UPLOAD_FOLDER'], event_original_filename)
            with open(filepath, 'rb') as src_file, open(display_original, 'wb') as dst_file:
                dst_file.write(src_file.read())
            
            # Create visualization and save with event-based name
            vis_filepath_web = os.path.join(app.config['UPLOAD_FOLDER'], event_result_filename)
            vis_filepath_output = os.path.join(app.config['OUTPUT_FOLDER'], event_result_filename)
            
            # Create output directory for visualization if it doesn't exist
            os.makedirs(os.path.dirname(vis_filepath_output), exist_ok=True)
            
            # Save the visualization
            vis_path = save_prediction_visualization(
                filepath, all_preds, all_confs, CLASS_NAMES,
                output_path=vis_filepath_web
            )
            
            # Also save a copy to the output folder
            if os.path.exists(vis_filepath_web):
                with open(vis_filepath_web, 'rb') as src_file, open(vis_filepath_output, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            
            # Send WhatsApp alert if class is in alert list
            alert_sent = False
            alert_message = "The detected event is not in your alert list."
            
            if event_name in alert_classes:
                success, message_id = send_whatsapp_alert(
                    phone_number,
                    event_name,
                    confidence * 100
                )
                alert_sent = success
                if not success:
                    alert_message = f"Error sending alert: {message_id}"
            
            # Pass just the filenames, not the relative paths
            return render_template(
                'result.html',
                original_video=event_original_filename,
                prediction_video=event_result_filename,
                predicted_class=event_name,
                confidence=confidence,
                processing_time=processing_time,
                class_counts=class_counts,
                alert_sent=alert_sent,
                alert_message=alert_message,
                phone_number=phone_number
            )
        else:
            flash('Could not process video. Please try another one.', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error processing video: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))
    finally:
        # Cleanup temporary file if needed
        if os.path.exists(filepath) and filename.startswith(str(uuid.UUID(int=0))[0:8]):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic video prediction"""
    global model
    # Check if model is loaded
    if model is None:
        model = load_detection_model()  # Try to load model
        if model is None:  # If still None, return error
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file provided'
        }), 400
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    # Get alert settings from form or JSON data - use fixed number
    data = request.form.to_dict() if request.form else request.get_json(silent=True) or {}
    alert_classes = data.get('alert_classes', '').split(',') if isinstance(data.get('alert_classes'), str) else data.get('alert_classes', [])
    phone_number = app.config['FIXED_RECEIVER_NUMBER']  # Always use fixed number
    
    # If no classes selected, use defaults
    if not alert_classes:
        alert_classes = DEFAULT_ALERT_CLASSES
    
    # Check if file type is allowed
    if file and allowed_file(file.filename):
        temp_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(filepath)
        
        try:
            # Process the video
            pred_class, confidence, all_preds, all_confs = predict_video(
                model, filepath, clip_len=16, device=device
            )
            
            if pred_class is not None:
                # Create prediction distribution
                class_distribution = {}
                for i, count in enumerate(torch.bincount(torch.tensor(all_preds), minlength=len(CLASS_NAMES))):
                    if count > 0:
                        class_distribution[CLASS_NAMES[i]] = {
                            'count': count.item(),
                            'percentage': count.item() / len(all_preds) * 100
                        }
                
                # Get the predicted event name
                event_name = CLASS_NAMES[pred_class]
                
                # Generate event-based filenames
                event_original_filename = generate_event_filename(file.filename, event_name, "original_")
                event_result_filename = generate_event_filename(file.filename, event_name, "result_")
                
                # Save files to output directory with event-based names
                original_event_path = os.path.join(app.config['OUTPUT_FOLDER'], event_original_filename)
                with open(filepath, 'rb') as src_file, open(original_event_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
                
                # Create visualization and save with event-based name
                vis_filepath = os.path.join(app.config['OUTPUT_FOLDER'], event_result_filename)
                
                # Create output directory for visualization if it doesn't exist
                os.makedirs(os.path.dirname(vis_filepath), exist_ok=True)
                
                # Save the visualization
                vis_path = save_prediction_visualization(
                    filepath, all_preds, all_confs, CLASS_NAMES,
                    output_path=vis_filepath
                )
                
                # Send WhatsApp alert if class is in alert list
                alert_sent = False
                alert_message = "The detected event is not in the alert list."
                alert_message_id = None
                
                if event_name in alert_classes:
                    success, message_id = send_whatsapp_alert(
                        phone_number,
                        event_name,
                        confidence * 100
                    )
                    alert_sent = success
                    alert_message_id = message_id
                    if not success:
                        alert_message = f"Error sending alert: {message_id}"
                
                # Create response
                response = {
                    'success': True,
                    'prediction': {
                        'class': event_name,
                        'class_id': int(pred_class),
                        'confidence': float(confidence),
                        'distribution': class_distribution
                    },
                    'files': {
                        'original': event_original_filename,
                        'result': event_result_filename,
                        'path': app.config['OUTPUT_FOLDER']
                    },
                    'alert': {
                        'sent': alert_sent,
                        'message': alert_message if not alert_sent else "Alert sent successfully",
                        'message_id': alert_message_id,
                        'phone': phone_number
                    }
                }
                
                return jsonify(response), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not process video'
                }), 500
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        finally:
            # Clean up the temporary file
            try:
                os.remove(filepath)
            except:
                pass
    
    return jsonify({
        'success': False,
        'error': 'File type not allowed'
    }), 400

# Add a special route to serve files from the static/uploads directory
@app.route('/static/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Load the model at startup
    model = load_detection_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)