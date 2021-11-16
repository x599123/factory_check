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
import multiprocessing as mp
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from vidgear.gears import NetGear
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, set_logging, increment_path,crop_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from PIL import Image
import cv2
import os
import numpy as np
import argparse
import torch
import sys
from PIL import Image, ImageDraw, ImageFont
from model import *
import argparse
import time

from flask import Flask, Response
import websocket
import ssl
import time
import json
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         '.','-'
        ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

web_frame=''
web_text=''
raw_frame=''

frame_temp=np.array([])
def get_frame():
    global frame_temp
    options = {"multiclient_mode": True}
    client = NetGear(
    address="127.0.0.1",
    port="5577",
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
            ws.connect("wss://172.16.1.221:3010/ws/admin-ai/1/")

            while True:
                
                #result = ws.recv()
                if web_text !='':
                
                                    
                    ws.send(json.dumps({
                    "data": {
                        "to_id": "admin-hand",
                        "meter_value": str(web_text)                    },
                    "eventName": "__ai_meter_result_value"
                    }))
                    print("websocket_send",web_text)
                #result = json.loads(result)
                #print("Received '%s'" % result)
                #ws.close()
                time.sleep(0.3)
        except Exception as ex:
            print("exception: ", format(ex))

    try:
        listenForever()
    except Exception as e :
        print("Exception occured: ",e)


#def job():
#    global frame_temp
    
#    def on_error(ws, error):
#        print(error)

#    def on_close(ws):
#        print("### closed ###")

#    def on_open(ws):
#        def run(*args):
#            #global web_text
#            while True:
#                ws.send(json.dumps({
#                "data": {
#                    "to_id": "admin-hand",
#                    "meter_value": web_text
#                },
#                "eventName": "__ai_meter_result_value"
#            }))
#            print("thread terminating...")
#        thread.start_new_thread(run, ())
#    def on_message(ws, message):
#        print(message)
#    websocket.enableTrace(True)
#    ws = websocket.WebSocketApp("wss://172.16.1.221:3010/ws/admin-ai/1/",
#                              on_message = on_message,
#                              on_error = on_error,
#                              on_close = on_close)
   
#    ws.on_open = on_open
#    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
def web_view(frame_buffer,text_buffer,web_buffer):
       global web_frame,web_text,raw_frame
       app = Flask(__name__)
       global web_out_label
       def get_image():
           while True:
               ret, jpeg = cv2.imencode('.jpg', web_frame)
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
           return Response(one_image(webin), mimetype="multipart/x-mixed-replace; boundary=frame")
       @app.route("/shutdown")
       def shutdown():
           #exit()
           os.system('ps -ef | grep eval.py| grep -v grep | awk \'{print $2}\' | xargs kill -9')
           os.system('ps -ef | grep monitor.py| grep -v grep | awk \'{print $2}\' | xargs kill -9')
           return Response('shutdown')
  # If the path is a digit, parse it as a webcam index
       app.run('0.0.0.0',port=8887)
def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 

    return inp
def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def digit_reconize(frame_buffer,text_buffer):
    global web_frame,web_text
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','-','.']
    CHARS_reverse = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,'10':'.', '11':'.'}
    lprnet_up = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet_up.to(device)
    lprnet_up.load_state_dict(torch.load('digits_noequ_98.91.pth'))
    lprnet_up.eval()
    ans_list=[]
    while True:   
        localtime = time.asctime( time.localtime(time.time()) )
        if frame_buffer.get() is not None:
            result1=frame_buffer.get()
            upper_res = cv2.resize(result1, (94, 24), interpolation=cv2.INTER_CUBIC) 
            upper_res = (np.asarray(upper_res, 'float32')- 37.7) / 255.
            upper_res = torch.FloatTensor(upper_res).unsqueeze(0).unsqueeze(0).to(device) 
            preds_up = lprnet_up(upper_res)
            preds_up = preds_up.cpu().detach().numpy()
            labels_up, pred_labels_up = decode(preds_up, CHARS)
            pred_labels_up = [CHARS_reverse[f'{d}'] for d in pred_labels_up[0]]
            pred_labels_up = ''.join([str(d) for d in pred_labels_up])
            output = cv2ImgAddText(frame_buffer.get(), pred_labels_up, (0, 10), textColor=(0,255,0), textSize=20)
            if '..' not in pred_labels_up and '.'  in pred_labels_up and not pred_labels_up.startswith('.') and (len(pred_labels_up)==4 or len(pred_labels_up)==5):
                try:
                    
                    ans = pred_labels_up.split('.')
                    if len(ans[1])==1:
                        ans_list.append(pred_labels_up)                      
                        if len(ans_list)==2:
                            result = ans_list.count(ans_list[0]) == len(ans_list)                    
                            if (result):
                                #print(float(ans_list[0]))                                
                                web_text=float(ans_list[0])
                               # print(web_text)
                                print(f'辨識結束:{web_text},{localtime}')

                            ans_list=[]
                                 

                except KeyboardInterrupt:
                    print('\nStopping...')
                    cleanup_and_exit()
                except:
                    web_text=''
                    pass
                    
            web_buffer.put(output)
            web_frame=output
        else:
            web_text=''

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels      = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        # torch.Size([CHARS length: 14, output length: 18 ])
        pred_label            = list()
        no_repeat_blank_label = list()
        
        # greedy decode here
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        blank = CHARS_DICT['-']
        output= []
        for i, d in enumerate(pred_label):
            if d == blank:
                output.append(d)
            else:
                if pred_label[i] == pred_label[i+1:i+2]:
                    pass
                else:
                    output.append(d)
        # 2. remove the blank
        output = [d for d in output if d != blank]
        pred_labels.append(output)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, pred_labels

@torch.no_grad()
def run(weights='first_digits.pt',  # model.pt path(s)
        weights2='second_digits.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.65,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        tfl_int8=False,  # INT8 quantized TFLite model
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    #ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
    #ws.connect("wss://172.16.1.221:3010/ws/admin-ai/1/")
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model1 = attempt_load(weights2, map_location=device)  # load FP32 model
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
       # print('dataset',dataset.shape)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        #print('dataset',dataset.shape)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        model1(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    global frame_temp
    global web_text
    while True:
        
        #img=frame_temp
        #cv2.imshow('aaa',img)

        if frame_temp is not None:
            if len(frame_temp)==0:
                continue
        else:
            web_text=''
            continue
            
        img=frame_temp
        im0s=frame_temp
        img = letterbox(img, 640, stride=stride, auto=True)[0]
        #img = np.stack(img, 0)
        #img=output_imgY
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)  
        #print(img.shape)


        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
            
        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                s, im0, frame =  '', im0s.copy(), getattr(dataset, 'frame', 0)

            #p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0,1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            global raw_frame
            raw_frame=im0
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
                    first_digit_image=crop_one_box(xyxy, imc, file='save.jpg'  ,BGR=True)
            else:
                web_text=''            
            # Print time (inference + NMS)
            if len(det):
            #stride, names = 64, [f'class{i}' for i in range(1000)] 
                im0=first_digit_image
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
            
                # Second-stage classifier (optional)
                if classify:
                    pred = apply_classifier(pred, modelc, img, first_digit_image)
                localtime = time.asctime( time.localtime(time.time()) )
                print("辨識開始:", localtime)
                print(f'{s}Done. ({t2 - t1:.3f}s)')    
                # Process predictions
                for i, det in enumerate(pred):  # detections per image
                    

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
                            result1=crop_one_box(xyxy, imc,  BGR=True)
                            cv2.imwrite('bbbb.jpg',result1)
                            result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
                            frame_buffer.put(result1)

                        
                        #    ws.send(json.dumps({
                        #    "data": {
                        #        "to_id": "admin-hand",
                        #        "meter_value": web_text
                        #    },
                        #    "eventName": "__ai_meter_result_value"
                        #}))
                        #      #result = json.loads(result)
                        #      #print("Received '%s'" % result)
                        #    ws.close()
        #break
    print(f'Done. ({time.time() - t0:.3f}s)')
    
   


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    #print(opt.imgsz)
    #opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.imgsz=[640, 640]
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))



if __name__ == "__main__":
    opt = parse_opt()
    #main(opt)
    frame_buffer = mp.Queue()
    text_buffer = mp.Queue()
    web_buffer  = mp.Queue()
    added_thread1 = threading.Thread(target=main,args=(opt,))
    added_thread1.start()
    added_thread2 = threading.Thread(target=digit_reconize,args=(frame_buffer, text_buffer,))
    added_thread2.start()
    added_thread = threading.Thread(target=web_view,name='t1',args=(frame_buffer, text_buffer,web_buffer,))
    added_thread.start()
    added_thread3 = threading.Thread(target=get_frame,args=())
    added_thread3.start()
    added_thread4 = threading.Thread(target=job,args=())
    added_thread4.start()
    added_thread.join()
    added_thread2.join()
    added_thread1.join()
    added_thread4.join()
