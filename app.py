import os
import uuid
import cv2
import subprocess
import shutil
import time
import json
from flask import Flask, render_template, request, redirect, url_for, Response, flash, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'webm'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in app.config['ALLOWED_IMAGE_EXTENSIONS'].union(app.config['ALLOWED_VIDEO_EXTENSIONS'])

def get_ffmpeg_path():
    """Find FFmpeg executable with platform-specific paths"""
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    if os.name == 'nt':  # Windows
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    
    return None

def process_detection_results(results, processing_time):
    """Extract statistics from detection results"""
    stats = {
        'time': {
            'preprocess': results[0].speed['preprocess'],
            'inference': results[0].speed['inference'],
            'postprocess': results[0].speed['postprocess'],
            'total': processing_time
        },
        'objects': {},
        'resolution': f"{results[0].orig_shape[1]}x{results[0].orig_shape[0]}",
        'fps': 0
    }
    
    # Count objects
    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls)]
            stats['objects'][cls] = stats['objects'].get(cls, 0) + 1
    
    return stats

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Unsupported file type', 'error')
        return redirect(url_for('index'))
    
    try:
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        unique_id = uuid.uuid4().hex
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.{ext}")
        result_path = os.path.join(app.config['RESULT_FOLDER'], f"{unique_id}.{ext}")
        
        file.save(input_path)
        
        if ext in app.config['ALLOWED_IMAGE_EXTENSIONS']:
            # Image processing
            start_time = time.time()
            results = model(input_path)
            processing_time = (time.time() - start_time) * 1000
            
            if not results or len(results) == 0:
                raise Exception("No detection results returned")
            
            stats = process_detection_results(results, processing_time)
            results[0].save(filename=result_path)
            
            return redirect(url_for('results', 
                                 input=f"{unique_id}.{ext}", 
                                 output=f"{unique_id}.{ext}",
                                 stats=json.dumps(stats)))
        
        elif ext in app.config['ALLOWED_VIDEO_EXTENSIONS']:
            # Video processing
            ffmpeg_path = get_ffmpeg_path()
            if not ffmpeg_path:
                flash('FFmpeg not found. Please install FFmpeg', 'error')
                return redirect(url_for('index'))
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            stats = {
                'time': {'total': 0, 'preprocess': 0, 'inference': 0, 'postprocess': 0},
                'objects': {},
                'resolution': f"{width}x{height}",
                'fps': fps
            }
            
            temp_output = os.path.join(app.config['RESULT_FOLDER'], f"temp_{unique_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                results = model(frame)
                frame_time = (time.time() - frame_start) * 1000
                
                stats['time']['total'] += frame_time
                stats['time']['inference'] += results[0].speed['inference']
                stats['time']['preprocess'] += results[0].speed['preprocess']
                stats['time']['postprocess'] += results[0].speed['postprocess']
                frame_count += 1
                
                for result in results:
                    for box in result.boxes:
                        cls = result.names[int(box.cls)]
                        stats['objects'][cls] = stats['objects'].get(cls, 0) + 1
                
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
            
            cap.release()
            out.release()
            
            # Calculate averages
            if frame_count > 0:
                for key in ['preprocess', 'inference', 'postprocess']:
                    stats['time'][key] /= frame_count
            
            # Convert with FFmpeg
            final_output = os.path.join(app.config['RESULT_FOLDER'], f"{unique_id}.mp4")
            ffmpeg_command = [
                ffmpeg_path,
                '-y', '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-loglevel', 'error',
                final_output
            ]
            
            subprocess.run(ffmpeg_command, check=True)
            os.remove(temp_output)
            
            return redirect(url_for('results',
                                 input=f"{unique_id}.{ext}",
                                 output=f"{unique_id}.mp4",
                                 stats=json.dumps(stats)))
    
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        flash(f'Processing error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    input_file = request.args.get('input')
    output_file = request.args.get('output')
    stats = request.args.get('stats', '{}')
    
    try:
        stats = json.loads(stats)
    except json.JSONDecodeError:
        stats = {
            'time': {'total': 0, 'preprocess': 0, 'inference': 0, 'postprocess': 0},
            'objects': {},
            'resolution': 'Unknown',
            'fps': 0
        }
    
    # Verify files exist
    input_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], input_file)) if input_file else False
    output_exists = os.path.exists(os.path.join(app.config['RESULT_FOLDER'], output_file)) if output_file else False
    
    return render_template('results.html',
                         input_url=url_for('static', filename=f'uploads/{input_file}') if input_exists else None,
                         output_url=url_for('static', filename=f'results/{output_file}') if output_exists else None,
                         stats=stats)

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)