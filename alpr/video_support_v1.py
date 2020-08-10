import sys, os
import keras
import cv2
import traceback
import numpy as np
import darknet.python.darknet as dn
from os.path import splitext, basename
from glob import glob
from darknet.python.darknet import detect
from src.label import dknet_label_conversion
from src.utils import nms
from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes

def adjust_pts(pts,lroi):
    return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

if __name__ == '__main__':

    try:
        cap = cv2.VideoCapture('demo.mp4')
        output_dir = 'lp_images/'
        i = 0

        lp_threshold = .5
        wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'
        wpod_net = load_model(wpod_net_path)
        ocr_threshold = .4
        ocr_weights = 'data/ocr/ocr-net.weights'
        ocr_netcfg  = 'data/ocr/ocr-net.cfg'
        ocr_dataset = 'data/ocr/ocr-net.data'
        ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
        ocr_meta = dn.load_meta(ocr_dataset)

        while(cap.isOpened()):
            ret, frame = cap.read()
            i += 1
            if i<500:
                continue
            w = frame.shape[0]
            h = frame.shape[1]
            ratio = float(max(frame.shape[:2]))/min(frame.shape[:2])
            side  = int(ratio*288.)
            bound_dim = min(side + (side%(2**4)),608)

            if i%10 == 0:
                Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(frame),bound_dim,2**4,(240,80),lp_threshold)
                cv2.imshow('detected_plate', frame)
                if len(LlpImgs):
                        Ilp = LlpImgs[0]
                        s = Shape(Llp[0].pts)
                        for shape in [s]:
                            ptsarray = shape.pts.flatten()
                            try:
                                frame = cv2.rectangle(frame,(int(ptsarray[0]*h), int(ptsarray[5]*w)),(int(ptsarray[1]*h),int(ptsarray[6]*w)),(0,255,0),3)
                                cv2.imshow('detected_plate', frame)
                            except:
                                traceback.print_exc()
                                sys.exit(1)
                        cv2.imwrite('%s/_lp.png' % (output_dir),Ilp*255.)
                        cv2.imshow('lp_bic', Ilp)
                        R,(width,height) = detect(ocr_net, ocr_meta, 'lp_images/_lp.png' ,thresh=ocr_threshold, nms=None)
                        if len(R):

                                L = dknet_label_conversion(R,width,height)
                                L = nms(L,.45)

                                L.sort(key=lambda x: x.tl()[0])
                                lp_str = ''.join([chr(l.cl()) for l in L])
                                print("License Plate Detected: ", lp_str)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)