import cv2
from flask import Flask, Response, request, jsonify
from flask_cors import CORS 
import time

app = Flask(__name__)
CORS(app)
HOST_IP = '0.0.0.0'
PORT = 8081  # Changed from 8080 to avoid conflicts


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)


if not camera.isOpened():
    print("Error: Could not open camera.")

current_state = {
    'up': False,
    'down': False,
    'left': False,
    'right': False,
    'command': 'stop'
}

def generate_frames():
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break

        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/control/set', methods=['POST'])
def control_set():
    global current_state
    
    try:
        data = request.get_json()
        current_state.update(data)

        print(f"Received Command: {current_state['command']}, State: {data}")
        return jsonify({'status': 'success', 'state': current_state})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    print(f"Streaming server running on http://{HOST_IP}:{PORT}")
    try:
        app.run(host=HOST_IP, port=PORT, threaded=True)
    finally:
        # Release the camera resource when the server is stopped
        camera.release()
