import cv2
import zmq
import base64
import numpy as np
from base_camera import BaseCamera

class Client(BaseCamera):
    def __init__(self):
        super(Client, self).__init__()
    
    @staticmethod
    def frames():
        context = zmq.Context()
        footage_socket = context.socket(zmq.SUB)
        footage_socket.bind('tcp://*:5555')
        footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        while True:
            try:
                frame = footage_socket.recv_string()
                img = base64.b64decode(frame)
                npimg = np.fromstring(img, dtype=np.uint8)
                source = cv2.imdecode(npimg, 1)
                
                _, jpeg = cv2.imencode('.jpg', source)
                frame1 =  jpeg.tobytes()
                yield frame1

            except Exception as e:
                print(e)