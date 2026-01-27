import cv2
from flask import Flask, Response
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera. Trying camera index 1...")
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("Still cannot open camera")
    else:
        print("Camera opened on index 1!")
else:
    print("Camera opened on index 0!")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

@app.route('/')
def home():
    return """
    <h1>Camera Server Running</h1>
    <p>Camera feed available at: <a href="/video_feed">/video_feed</a></p>
    <img src="/video_feed" width="640">
    """

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    PORT = 8081
    print(f"Camera server running on http://localhost:{PORT}")
    print(f"Open browser to: http://localhost:{PORT}")
    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True, debug=True)
    finally:
        camera.release()
        print("Camera released")
