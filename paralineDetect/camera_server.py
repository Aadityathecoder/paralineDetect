import cv2
import numpy as np
from flask import Flask, Response
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Try different camera indices
camera = None
for index in [0, 1, 2]:
    print(f"Trying camera index {index}...")
    cam = cv2.VideoCapture(index)
    if cam.isOpened():
        ret, frame = cam.read()
        if ret and frame is not None:
            print(f"SUCCESS! Camera {index} is working!")
            camera = cam
            break
        else:
            print(f"Camera {index} opened but can't read frames")
            cam.release()
    else:
        print(f"Camera {index} failed to open")

if camera is None:
    print("ERROR: No working camera found!")
    exit(1)

# Set camera properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_blue_lanes(frame):
    """Apply blue lane detection to a frame"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color (blue tape)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame
    result = frame.copy()
    for contour in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(contour) < 100:
            continue
        
        # Draw the contour with green outline
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
    
    return result

def generate_frames_raw():
    """Generate raw camera frames"""
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames_processed():
    """Generate frames with lane detection"""
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break
        
        # Apply lane detection
        processed_frame = detect_blue_lanes(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Failed to encode frame")
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Raw video feed (for Quadrant 3)"""
    return Response(generate_frames_raw(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_processed')
def video_feed_processed():
    """Processed video feed with lane detection (for Quadrant 1)"""
    return Response(generate_frames_processed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return """
    <h1>Camera Server Running</h1>
    <p>Raw feed: <a href='/video_feed'>/video_feed</a></p>
    <p>Lane detection feed: <a href='/video_feed_processed'>/video_feed_processed</a></p>
    """

if __name__ == '__main__':
    print("\nCamera server starting on http://localhost:8081")
    print("Raw feed at: http://localhost:8081/video_feed")
    print("Lane detection feed at: http://localhost:8081/video_feed_processed\n")
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        camera.release()
        print("Camera released")
