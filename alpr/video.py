import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

import base64
import zmq

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir, isfile
from os 					import makedirs
from src.label				import dknet_label_conversion, lread, Label, readShapes
from src.utils 				import crop_region, image_files_from_folder
from src.drawing_utils			import draw_label, draw_losangle, write2img
from darknet.python.darknet import detect

import sys, os
import keras
import cv2
import traceback
from pdb import set_trace as pause

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
from src.utils 				import nms

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

x = cv2.VideoCapture('demo.mp4')
i = 0

context= zmq.Context()
footage_socket= context.socket(zmq.PUB)
footage_socket.connect('tcp://localhost:5555')

vehicle_threshold = .5
vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
vehicle_dataset = 'data/vehicle-detector/voc.data'
vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)

lp_threshold = .5
wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
wpod_net = load_model(wpod_net_path)
lp_dir = 'lp_images/'

ocr_threshold = .4
ocr_weights = 'data/ocr/ocr-net.weights'
ocr_netcfg  = 'data/ocr/ocr-net.cfg'
ocr_dataset = 'data/ocr/ocr-net.data'
ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)

track_file = open('track.txt', "a")

f_count = 0
while True:

    # input_dir = 'test_input/'
    # output_dir = 'vl_images/'
    input_dir = 'test_input/'
    output_dir = 'test_output'

    _, frame = x.read()
    # frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25, interpolation=cv2.INTER_AREA)
    f_count += 1
    if f_count < 500 or f_count%10 != 0:
        continue

    filename = input_dir + 'image' + '.png'
    # print(filename)

    cv2.imwrite(filename, frame)
    imgs_paths = image_files_from_folder(input_dir)
    imgs_paths.sort()

    print 'Searching for vehicles using YOLO...'

    for i,img_path in enumerate(imgs_paths):
        print '\tScanning %s' % img_path

        bname = basename(splitext(img_path)[0])

        R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)

        R = [r for r in R if r[0] in ['car','bus']]

        print '\t\t%d cars found' % len(R)

        if len(R):

            Iorig = cv2.imread(img_path)
            WH = np.array(Iorig.shape[1::-1],dtype=float)
            Lcars = []

            for iterator,r in enumerate(R):

                cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                tl = np.array([cx - w/2., cy - h/2.])
                br = np.array([cx + w/2., cy + h/2.])
                label = Label(0,tl,br)
                Icar = crop_region(Iorig,label)

                Lcars.append(label)

                print "Generating by vehicle"

                cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,iterator),Icar)

            lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)
    
    print 'Searching for plates using YOLO...'

    # input_dir = 'vl_images/'
    # output_dir = 'lp_images/'

    input_dir = 'test_output/'
    output_dir = input_dir
    
    imgs_paths = glob('%s/*car.png' % input_dir)

    for i,img_path in enumerate(imgs_paths):
        print '\t Processing %s' % img_path

        bname = splitext(basename(img_path))[0]
        Ivehicle = cv2.imread(img_path)

        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
        print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

        Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)

            print "Generating by lp"

            cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
            cv2.imwrite('%s/%s_lp.png' % (lp_dir,bname),Ilp*255.)
            writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

    input_dir = 'test_output/'
    output_dir = input_dir
    
    imgs_paths = sorted(glob('%s/*lp.png' % input_dir))

    print 'Performing OCR...'

    for i,img_path in enumerate(imgs_paths):

        print '\tScanning %s' % img_path

        bname = basename(splitext(img_path)[0])
        R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

        if len(R):
            L = dknet_label_conversion(R,width,height)
            L = nms(L,.45)
            L.sort(key=lambda x: x.tl()[0])
            lp_str = ''.join([chr(l.cl()) for l in L])

            print "Generating by ocr"

            with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
                f.write(lp_str + '\n')
            print '\t\tLP: %s' % lp_str

        else:

            print 'No characters found'
    
    GREEN = (  0, 255, 0)
    RED    = (  0,  0,255)

    input_dir = 'test_input/'
    output_dir = 'test_output/'

    img_files = image_files_from_folder(input_dir)

    for img_file in img_files:

        bname = splitext(basename(img_file))[0]

        I = cv2.imread(img_file)

        detected_cars_labels = '%s/%s_cars.txt' % (output_dir,bname)

        Lcar = lread(detected_cars_labels)

        track_file.write(str(f_count))

        if Lcar:

            for i,lcar in enumerate(Lcar):

                draw_label(I,lcar,color=GREEN,thickness=3)

                lp_label 		= '%s/%s_%dcar_lp.txt'		% (output_dir,bname,i)
                lp_label_str 	= '%s/%s_%dcar_lp_str.txt'	% (output_dir,bname,i)

                if isfile(lp_label):

                    Llp_shapes = readShapes(lp_label)
                    pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                    ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                    draw_losangle(I,ptspx,RED,3)

                    if isfile(lp_label_str):
                        with open(lp_label_str,'r') as f:
                            lp_str = f.read().strip()
                        llp = Label(0,tl=pts.min(1),br=pts.max(1))
                        write2img(I,llp,lp_str)

                        # sys.stdout.write(',%s' % lp_str)
                        # track_file = open('track.txt', "a")
                        track_file.write(',%s' % lp_str)

        cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
        # sys.stdout.write('\n')
        # track_file = open('track.txt', "a")
        track_file.write('\n')
    reads = cv2.imread('test_output/image_output.png')
    _, buffer = cv2.imencode('.jpg', reads)
    jpg_as_text = base64.b64encode(buffer)
    footage_socket.send(jpg_as_text)

    for file in os.listdir('test_output'):
        if file.endswith('_lp.png') or file.endswith('car.png') or file.endswith('_cars.txt') or file.endswith('_lp.txt') or file.endswith('_str.txt'):
            os.remove('test_output/' + file)