# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
import threading
from queue import Queue
import multiprocessing as mp
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,crop_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import sys
from PIL import Image, ImageDraw, ImageFont
from model import *
import argparse
import time
from multiprocessing import Process, Pool
from flask import Flask, render_template, Response
from vidgear.gears import NetGear
import websocket,ssl,json
web_frame=''
raw_frame=''
frame_temp=np.array([])
web_text=''
def get_frame():
    global frame_temp
    options = {"multiclient_mode": True}
    client = NetGear(
    address="127.0.0.1",
    port="5599",
    protocol="udp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
) 
    while True:
        frame= client.recv()
        frame_temp=frame


def job():
    global web_text
    def listenForever():
        try:
            # ws = create_connection("wss://localhost:9080/websocket")
            ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
            ws.connect("wss://127.0.0.1:3010/ws/admin-ai/1/")

            while True:
                
                #result = ws.recv()
                if web_text !='' and type(web_text)==str:
                    ws.send(json.dumps({
                    "data": {
                        "to_id": "e000099999-hand",
                        "meter_value": web_text
                    },
                    "eventName": "__ai_meter_result_value"
                    }))
                    print("websocket_send",web_text)
                #result = json.loads(result)
                #print("Received '%s'" % result)
                #ws.close()
                time.sleep(1)
        except Exception as ex:
            print("exception: ", format(ex))

    try:
        listenForever()
    except Exception as e :
        print("Exception occured: ",e)
def web_view(web_buffer):
       global web_frame
       global raw_frame
       global web_text
       app = Flask(__name__)
       def get_image():
           while True:
               ret, jpeg = cv2.imencode('.jpg', raw_frame)
               jpeg = jpeg.tobytes()
               yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+ jpeg + b'\r\n')
               time.sleep(0.03) # my Firefox needs some time to display image / Chrome displays image without it
       @app.route("/")
       def stream():
           return Response(get_image(), mimetype="multipart/x-mixed-replace; boundary=frame")
       def one_image():
           while True:
               ret, jpeg = cv2.imencode('.jpg', raw_frame)
               jpeg = jpeg.tobytes()
               yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+ jpeg + b'\r\n')
               time.sleep(0.03) # my Firefox needs some time to display image / Chrome displays image without it
       @app.route("/stream")
       def stream1():
           return Response(one_image(), mimetype="multipart/x-mixed-replace; boundary=frame")

       @app.route("/out")
       def out():
           return Response(str(web_text), mimetype='text/html') 
               
       @app.route("/in")
       def webin():
           return Response(one_image(web_in), mimetype="multipart/x-mixed-replace; boundary=frame")
       @app.route("/shutdown")
       def shutdown():
           #exit()
           os.system('ps -ef | grep eval.py| grep -v grep | awk \'{print $2}\' | xargs kill -9')
           os.system('ps -ef | grep monitor.py| grep -v grep | awk \'{print $2}\' | xargs kill -9')
           return Response('shutdown')
  # If the path is a digit, parse it as a webcam index
       app.run('0.0.0.0',port=8888)

@torch.no_grad()
def run(weights='gauge_weight.pt',  # model.pt path(s)
        weights2='niddle_weight.pt',
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        #project='runs/detect',  # save results to project/name
        #name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        tfl_int8=False,  # INT8 quantized TFLite model
        ):
    global web_frame
    global web_text
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        model1 = attempt_load(weights2, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        model1(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    while True:
        localtime = time.asctime( time.localtime(time.time()) )
        #print("辨識開始:", localtime)
        #if len(frame_temp)==0:
        #    continue
        img=frame_temp
        im0s=frame_temp
        img = letterbox(img, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)  
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
           # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if tfl_int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if tfl_int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                 s, im0, frame =  '', im0s.copy(), getattr(dataset, 'frame', 0)

             # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            global raw_frame
            
            if len(det):
                web_text1=[]
                temp_list_class_cor=[]
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}, "  # add to string

                #web_text=s
                #print(web_text)
                # Write results
                temp_list_class_cor=[]
                for *xyxy, conf, cls in reversed(det):
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        first_digit_image=crop_one_box(xyxy, imc, file='save.jpg'  ,BGR=True)
                        cv2.imwrite('bbbb.jpg',first_digit_image)
                        xywh=(xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        cor=xywh[:2]
                im0=first_digit_image
                first_plot=im0
                img = letterbox(first_digit_image, 640, stride=stride, auto=False)[0]

                #img = np.stack(img, 0)
                #img=output_img
                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)  
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float() 
                img = img / 255.0          
                if len(img.shape) == 3:
                    img = img[None]
                pred = model1(img, augment=augment, visualize=visualize)[0]
                
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                t2 = time_sync()
                for i, det in enumerate(pred):  # detections per image
                    
                    names = model1.module.names if hasattr(model, 'module') else model1.names  # get class names
                    print(names)
                    #p, s, im0, frame = path[i], f'{i}: ', first_digit_image.copy(), getattr(dataset, 'frame', 0)                    
                    #p = Path(p)  # to Path
                    #save_path = str(save_dir / p.name)  # img.jpg
                    #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0,1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    if len(det):
                    
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            result1=crop_one_box(xyxy, imc,  BGR=True)
                            box_img = plot_one_box(xyxy, np.ascontiguousarray(imc), label=label, color=colors(c, True), line_width=line_thickness)
                            
                            result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
                            raw_frame=box_img
                print(f'{s}Done.)')

            else:
                web_text=''  
                        #print([chinese_names[int(cls)]]+cor)
                #print(f'{s}Done. ({t2 - t1:.3f}s)')
                #print(web_text1)

                #print(f'"辨識結束:"{web_text}{localtime}({t2 - t1:.3f}s)')
                #break
                #print(web_text1)
                    #if web_frame == None:
                    #     raw_frame
                        #if save_crop:
                        #    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        #break
            # Print time (inference + NMS)
            

            # Stream results
            #if view_img:
            #    cv2.imshow(str(p), im0)
            #    cv2.waitKey(1)  # 1 millisecond

            ## Save results (image with detections)
            #if save_img:
            #    if dataset.mode == 'image':
            #        cv2.imwrite(save_path, im0)
            #    else:  # 'video' or 'stream'
            #        if vid_path[i] != save_path:  # new video
            #            vid_path[i] = save_path
            #            if isinstance(vid_writer[i], cv2.VideoWriter):
            #                vid_writer[i].release()  # release previous video writer
            #            if vid_cap:  # video
            #                fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #            else:  # stream
            #                fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                save_path += '.mp4'
            #            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #        vid_writer[i].write(im0)

    #if save_txt or save_img:
    #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #    print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    opt.imgsz = [640, 640]  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    #main(opt)
    web_buffer  = mp.Queue()
    added_thread1 = threading.Thread(target=main,args=(opt,))
    added_thread1.start()
    added_thread = threading.Thread(target=web_view,name='T1',args=(web_buffer,))
    added_thread.start()
    added_thread2 = threading.Thread(target=get_frame,args=())
    added_thread2.start()    
    added_thread3 = threading.Thread(target=job,args=())
    added_thread3.start()
    
    added_thread.join()
    added_thread1.join()
    added_thread2.join()
    added_thread3.join()
