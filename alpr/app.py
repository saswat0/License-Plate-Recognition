import json
import cv2
import zmq
import base64
import numpy as np
from client import Client
from flask import Flask, render_template, Response, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
    
def gen(client):
    """Video streaming generator function."""
    while True:
        frame = client.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
    
def gen(client):
    """Video streaming generator function."""
    while True:
        frame = client.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    
    return Response(gen(Client()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
