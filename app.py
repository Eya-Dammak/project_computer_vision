from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
from fruit_detector import FruitDetector
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

# Initialize detector
detector = FruitDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save uploaded file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check file type
            extension = file.filename.lower().split('.')[-1]
            
            if extension in {'png', 'jpg', 'jpeg', 'gif'}:
                # Process image
                img = cv2.imread(filepath)
                processed_img, fruit_data, fruit_counts = detector.detect_fruits(img)
                
                # Save processed image
                processed_filename = f"processed_{filename}"
                processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, processed_img)
                
                return render_template('results.html', 
                                    original=filename,
                                    processed=processed_filename,
                                    fruit_data=fruit_data,
                                    fruit_counts=fruit_counts)
            else:
                # Process video
                return render_template('video.html', 
                                    video_file=filename)
    
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video_file = request.form['video_file']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{video_file}")
    
    # Process video
    detector.process_video(video_path, output_path)
    
    return {
        'status': 'completed',
        'processed_video': f"processed_{video_file}"
    }

@app.route('/')
def home():
    return "Fruit Detection System is Running!"

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug
        use_reloader=False  # Disable auto-reloader
    )