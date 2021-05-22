import sys
import os
import cv2
import zmq
import base64
import traceback
import numpy as np
import time
import darknet.python.darknet as dn
import common
import json
import argparse
from collections import Counter
from wpod_trt import get_engine
from src.utils import im2single, nms
from src.keras_utils import reconstruct
from src.label import Shape, writeShapes, dknet_label_conversion
from darknet.python.darknet import detect


def majority_voting(LP):
    dict = {}
    for index, L in enumerate(LP):
        for x in range(len(L)):
            key = x
            dict.setdefault(key, []).append(chr(L[x].cl()))
    dict1 = {}
    for q in range(len(dict.keys())):
        if len(dict[q]) > 3:
            dict1.setdefault(q, []).append(dict[q])
    lp = ''
    for x in range(len(dict1.keys())):
        res = Counter(dict1[x])
        lp += res.most_common(1)[0][0]
    return lp


def writeLP(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(content, f)


def sorting(L):  # distinguish one layer or two layer
    sum_h = []
    for l in L:
        sum_h.append(l.tl()[1])
    mean = np.mean(sum_h)
    var = np.var(sum_h)
    if var < 0.001:
        L.sort(key=lambda x: x.tl()[0])  # one layer
    else:
        L_top = []
        L_bot = []
        for l in L:
            if l.tl()[1] < mean:
                L_top.append(l)
            elif l.tl()[1] >= mean:
                L_bot.append(l)
        L_top.sort(key=lambda x: x.tl()[0])
        L_bot.sort(key=lambda x: x.tl()[0])
        L = L_top + L_bot
#L.sort(key=lambda x: x.tl()[0])
    lpstr = ''.join([chr(l.cl()) for l in L])
    return lpstr


def OCR_detection(img_path):
    R, (width, height) = detect(ocr_net, ocr_meta,
                                img_path.encode('utf-8'), thresh=ocr_threshold, nms=None)
    R1 = []
    if len(R):
        for r in R:
            if len(r[0]) < 2 and r[2][3] > 30:
                R1.append(r)
        L = dknet_label_conversion(R1, width, height)
        L = nms(L, .45)
        L.sort(key=lambda x: x.tl()[0])
        #LP_str = sorting(L)
        return L
    else:
        print('Characters not recognized')
        return False


def compute_wh(I):
    net_step = 2**4
    ratio = float(max(I.shape[:2]))/min(I.shape[:2])  # resizing ratio
    side = int(ratio*288.)
    bound_dim = min(side + (side % (2**4)), 608)
#    print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
    min_dim_img = min(I.shape[:2])
    factor = float(bound_dim)/min_dim_img
    w, h = (np.array(I.shape[1::-1], dtype=float)*factor).astype(int).tolist()
    w += (w % net_step != 0)*(net_step - w % net_step)
    h += (h % net_step != 0)*(net_step - h % net_step)
    return w, h


def LP_detection(image):
    Ivehicle = im2single(image)
    w, h = compute_wh(Ivehicle)

    Iresized = cv2.resize(Ivehicle, (768, 480))
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    inputs_alloc[0].host = T
    time_inference = time.clock()
    trt_outputs = common.do_inference(
        context, bindings=bindings, inputs=inputs_alloc, outputs=outputs_alloc, stream=stream)
    time_inference = time.clock() - time_inference
    trt_outputs = np.array(trt_outputs)
    trt_outputs = trt_outputs.reshape(30, 48, 8)
    Llp, LlpImgs = reconstruct(
        Ivehicle, Iresized, trt_outputs, (240, 80), lp_threshold)

    if len(LlpImgs):
        print('LP detected!')
        Ilp = LlpImgs[0]
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

        time_now = time.time()
        img_name = "%s%s.jpg" % (time.strftime(
            "%Y%m%d%H%M%S_", time.localtime(time_now)), str(time_now % 1.0)[2:4])
        img_path = os.path.join(output, img_name)
        cv2.imwrite(img_path, Ilp*255.)

        return img_path
    else:
        return False


def car_pass_by():
    c = 0
    freq = 2
    majorvotefreq = 0
    LP_6 = []
    cap = cv2.VideoCapture(avi)
    is_opened, frame = cap.read()
    while is_opened:
        if (c % freq == 0):
            F = LP_detection(frame)
            if (F == False):
                print('LP not found!')
            else:
                LP = OCR_detection(F)
                if LP == False:
                    pass
                else:
                    LP_6.append(LP)
                    majorvotefreq += 1
                    if majorvotefreq >= 7:
                        LPstr = majority_voting(LP_6)
                        LP_str = {'LP': LPstr}
                        time_now = time.time()
                        file_name = "%s%s.json" % (time.strftime(
                            "%Y%m%d%H%M%S_", time.localtime(time_now)), str(time_now % 1.0)[2:4])
                        file_path = os.path.join(output, file_name)
                        writeLP(file_path, LP)
                        print('/t/t LP:%s saved' % LP)
                        LP_6 = []
                        majorvotefreq = 0
        is_opened, frame = cap.read()
        c += 1

        if c >= 38:
            return -1


parser = argparse.ArgumentParser()
parser.add_argument('-lpth' 	, '--lp-threshold'	, type=float,
                    default=.5		, help='lp_threshold')
parser.add_argument('-ocrth' 	, '--ocr-threshold'	,
                    type=float, default=.4		, help='ocr_threshold')
parser.add_argument('-onnx'		, '--onnx-file'		, type=str,
                    default='models/wpod0309b.onnx', help='path to onnx')
parser.add_argument('-e'		, '--engine', type=str,
                    default='model0309b.engine'	, help='path to TRT engine')
parser.add_argument('-output'	, '--output'	    	, type=str,
                    default='samples/output'		, help='output file path')
parser.add_argument('-ocrw'		, '--ocr-weights'	, type=str,
                    default='data/ocr/yolov3-LP_40000.weights'		, help='OCR weights')
parser.add_argument('-ocrcfg'	, '--ocr-netcfg'		, type=str,
                    default='data/ocr/yolov3-LP.cfg_train'	, help='OCR cfg file')
parser.add_argument('-ocrdata'	, '--ocr-dataset'	, type=str,
                    default='data/ocr/LP.data'		, help='OCR dataset')
parser.add_argument('-a'		, '--avi'	, type=str,
                    default='test0305.mp4'		, help='avi path')
args = parser.parse_args()


if __name__ == '__main__':
    context = zmq.Context()
    footage_socket = context.socket(zmq.PUB)
    footage_socket.connect('tcp://localhost:5555')

    try:
        lp_threshold = args.lp_threshold
        ocr_threshold = args.ocr_threshold

        onnx_file_path = args.onnx_file
        engine_file_path = args.engine
        output = args.output

        ocr_weights = args.ocr_weights
        ocr_netcfg = args.ocr_netcfg
        ocr_dataset = args.ocr_dataset
        avi = args.avi
        ocr_net = dn.load_net(ocr_netcfg.encode('utf-8'),
                              ocr_weights.encode('utf-8'), 0)
        ocr_meta = dn.load_meta(ocr_dataset.encode('utf-8'))

        with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
            # because the input_size is fixed, the buffers should be allcated in advanced for all the data(img)
            inputs_alloc, outputs_alloc, bindings, stream = common.allocate_buffers(
                engine)

            c = 1
            freq = 5
            predetect_times = 0
            cap = cv2.VideoCapture(avi)
            is_opened, frame = cap.read()
            frame = cv2.resize(frame, (1920, 1200))
            while is_opened:
                if (c % freq == 0):
                    F = LP_detection(frame)
                    if (F == False):
                        print('LP not found!')
                    else:
                        LP = OCR_detection(F)
                        if LP == False:
                            pass
                        else:
                            predetect_times += 1
                            print('/t/t LP:%s' % LP)
                is_opened, frame = cap.read()
                c += 1
                if predetect_times >= 4:
                    LPreal = car_pass_by()
                    predetect_times = 0
                else:
                    predetect_times = 0

                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    jpg_as_text = base64.b64encode(buffer)
                    footage_socket.send(jpg_as_text)
                except Exception as e:
                    pass
    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
