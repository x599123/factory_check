import os, sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow import keras 
import pickle
import PIL.Image as Image
import random
import cv2
import time
import json
import multiprocessing as mp
# 選擇第二隻攝影機
import time
from flask import Flask, Response
import ssl
from vidgear.gears import NetGear
import socket
import numpy as np
import cv2
import pickle
import threading
import time
import websocket
try:
    import thread
except ImportError:
    import _thread as thread
frame_temp=np.array([])
patrol_point=''
socket_switch='off'
# 子執行緒的工作函數
def get_frame():
    options = {"multiclient_mode": True}
    client = NetGear(
    address="127.0.0.1",
    port="5567",
    protocol="udp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
) 
    global frame_temp



    while True:

        frame= client.recv()
        #print(frame.shape)
        frame=cv2.resize(frame,(128,128))
        frame = np.array(frame)/255
        frame = np.expand_dims(frame, axis=0)
        frame_temp=frame

        #print(frame_temp.shape)
            #print(len(indata))
            #img=np.asarray(bytearray(indata), dtype="uint8")

            #img=cv2.imdecode(img,3)
            #img=Image.fromarray(img, 'RGB')
            #img=img.resize((128,128))
            #img = np.array(img)/255
            #img = np.expand_dims(img, axis=0)
            #frame_temp.append(img)
            #cv2.imwrite('socket_get.jpg',img)
def web_view(socket_buff):
       
       app = Flask(__name__)
       global web_out_label
       @app.route("/on")
       def on():
           global socket_switch
           #exit()
           socket_switch='on'
           #socket_buff.put('on')

           return Response(socket_switch)
       @app.route("/off")
       def off():
           global socket_switch
           #exit()
           socket_switch="off"
           #socket_buff.put('off')
           
           return Response(socket_switch)
       @app.route("/status")
       def sta():
           #exit()
           
           #socket_buff.put('off')
           
           return Response(socket_switch)
  # If the path is a digit, parse it as a webcam index
       app.run('0.0.0.0',port=8886)
def job():
    global patrol_point,socket_switch
    def listenForever():
        try:
            # ws = create_connection("wss://localhost:9080/websocket")
            ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
            ws.connect("wss://172.16.1.221:3010/ws/admin-ai/1/")

            while True:
                #print(socket_switch)
                #result = ws.recv()
                if patrol_point !='' and patrol_point !='back' and socket_switch=='on':
                    ws.send(json.dumps({
                    "data": {
                        "to_id": "admin-hand",
                        "station_id": patrol_point
                    },
                    "eventName": "__ai_meter_result_poi"
                    }))
                   # print('websocket-send',patrol_point)
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

def classification():

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    model = keras.models.load_model('InceptionV3_patrol_new_withback.h5')
    global frame_temp
    global patrol_point
    count=0
    output_result=[]
    while(True):
        localtime = time.asctime( time.localtime(time.time()) )
        #time.sleep(0.5)
        frame=frame_temp
        #print(frame.shape)
        if len(frame)>0:
            #print(len(frame))

            #time.sleep(0.03)
            #print(count)
            result=model.predict(frame)
            #print(result)
            classes=['NY-HWS-0001','NY-TS-DM0002','NY-TS-DM0001','back']
            #print('the predicted type of img is: ', classes[np.argmax(result)])
            output_result.append(np.argmax(result))
            #print(pathetic_status)
            if len(output_result)==1:
                 print("辨識開始:", localtime)
            if len(output_result)==7:
                pathetic_status=output_result.count(output_result[0]) == len(output_result)
                if pathetic_status:
                    #print(output_result[0])
                    print('the output predicted type of img is: ', classes[output_result[0]])
                    patrol_point=classes[output_result[0]]
                    print(f'辨識結束:{localtime}')
                output_result.clear()
                    


socket_buff  = mp.Queue()

## 建立一個子執行緒
t = threading.Thread(target = get_frame)
# 執行該子執行緒
t.start()
s = threading.Thread(target = classification)
# 執行該子執行緒
s.start()
r = threading.Thread(target = job)
# 執行該子執行緒
r.start()
added_thread = threading.Thread(target=web_view,name='t1',args=(socket_buff,))
added_thread.start()
# 主執行緒繼續執行自己的工作


# 等待 t 這個子執行緒結束
t.join()
s.join()
r.join()
added_thread.join()
print("Done.")

