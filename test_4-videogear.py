import sys
import json
import asyncio
import argparse
import secrets
import ssl
import pathlib
import threading
import socket,pickle
import time
from typing import Optional, Dict, Awaitable, Any, TypeVar
from asyncio.futures import Future
import cv2
from pymediasoup import Device
from pymediasoup import AiortcHandler
from pymediasoup.transport import Transport
from pymediasoup.consumer import Consumer
from pymediasoup.producer import Producer
from pymediasoup.data_consumer import DataConsumer
from pymediasoup.data_producer import DataProducer
from pymediasoup.sctp_parameters import SctpStreamParameters

# Import aiortc
from aiortc import VideoStreamTrack
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack, MediaStreamError
from aiortc.contrib.media import MediaPlayer, MediaBlackhole, MediaRecorder, MediaRelay
from av import VideoFrame
import av

# Implement simple protoo client
import websockets
from random import random

import logging

from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import numpy as np 
upload_image=np.array([])
#logging.basicConfig(level=logging.DEBUG)





# Define NetGear Client at given IP address and assign list/tuple of
# all unique Server((5577,5578) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!


                                                                                                                                                                           

async def comsume_remote_track(track):
    global upload_image

    print("startComsuming")
    while True:
        try:
            
            frame = await track.recv()
            print("frame_received")
            img=frame.to_ndarray(format='bgr24')
            upload_image=img
            


            #if not (recv_data is None):
            ## extract unique port address and its respective data
            #    unique_address, data = recv_data
            ## update the extracted data in the data dictionary
            #    data_dict[unique_address] = data

            #if data_dict:
            ## print data just received from Client(s)
            #    for key, value in data_dict.items():
            #        print("Client at port {} said: {}".format(key, value))
            #result, imgencode = cv2.imencode('.jpg', img, encode_param)
            #stringData = imgencode.tostring()
            #writer.write(stringData)
            #print(img.shape)
            #cv2.imshow("Video", img)
            

              # 若按下 q 鍵則離開迴圈
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
                

            #ndarray = VideoFrame.to_ndarray()

            #await frameQueue.put(frame)
        
        except MediaStreamError as e:
            print(e)
            return
        except Exception as e:
            print(e)
            print(e.args)


####################################################################


class RemoteTrackReceiver(VideoStreamTrack):
    

    def __init__(self):
        super().__init__()
        self.kind = "video"
        self.frame = VideoFrame(width=640, height=480)
        self.task = None

    def startComsuming(self, track) :
        print("startComsuming")
        self.task = asyncio.ensure_future(comsume_remote_track(track))

    def stopComsuming(self) :
        if self.task is not None:
            self.task.cancel()
            self.task = None

    async def recv(self):
        timestamp, video_timestamp_base = await self.next_timestamp()
        """
        pts, time_base = await self.next_timestamp()
        print(type(self.track))

        frame = VideoFrame(width=640, height=480)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        return frame
        """
        #print("finalt")
        self.frame = VideoFrame(width=640, height=480)

        #while not frameQueue.empty(): 
            #self.frame = await frameQueue.get()
            #print("frame sended")


        self.frame.pts = timestamp
        self.frame.time_base = video_timestamp_base
        
        await asyncio.sleep(0.01)

        return self.frame


    
    
            

        


T = TypeVar("T")

class Demo():
    
    def __init__(self, uri, player=None, recorder=MediaBlackhole(), loop=None):
        if not loop:
            if sys.version_info.major == 3 and sys.version_info.minor == 6:
                loop = asyncio.get_event_loop()
            else:
                loop = asyncio.get_running_loop()
        self._loop = loop
        self._uri = uri
        self._player = player
        self._recorder = None
        self._recorder_dict = {}
        self._loopback_video_track = None
        self._new_loopback_video_track = None
        self._loopback_audio_track = None
        self._new_loopback_audio_track = None
        self.comsumer_changed = False
        # Save answers temporarily
        self._answers : Dict[str, Future] = {}
        self._websocket = None
        self._device = None

        self._tracks = []

        if player and player.audio:
            audioTrack = player.audio
        else:
            audioTrack = AudioStreamTrack()
        if player and player.video:
            videoTrack = player.video
        else:
            videoTrack = VideoStreamTrack()

        self._videoTrack = RemoteTrackReceiver()
        #self._audioTrack = audioTrack

        self._tracks.append(videoTrack)
        #self._tracks.append(audioTrack)

        self._sendTransport: Optional[Transport] = None
        self._recvTransport: Optional[Transport] = None

        self._producers = []
        self._consumers = []
        self._tasks = []
        self._consume_task = None
        self.users = {}
        self._closed = False
        self._blackholes = []

    # websocket receive task
    async def recv_msg_task(self):
        while True:
            await asyncio.sleep(0.01)
            if self._websocket != None:
                message = json.loads(await self._websocket.recv())
                if message.get('response'):
                    if message.get('id') != None:
                        self._answers[message.get('id')].set_result(message)
                elif message.get('request'):
                    if message.get('method') == 'newConsumer':
                        print("!!!!newConsumerMessage:", message)
                        
                        peerId = message['data']['peerId']
                        cname = message['data']['rtpParameters']['rtcp']['cname']
                        producerId = message['data']['producerId']
                        comsumerId = message['data']['id']
                        kind = message['data']['kind']

                        if peerId in self.users.keys() :
                            self.users[ peerId ]['cname'] = cname
                            self.users[ peerId ]['comsumers'].append({"producerId": producerId, "comsumerId": comsumerId, "kind": kind})
                        
                        print("!!!!current user state:", self.users)

                        await self.consume(
                            id=message['data']['id'],
                            producerId=message['data']['producerId'],
                            kind=message['data']['kind'],
                            rtpParameters=message['data']['rtpParameters'],
                            displayName= self.users[ peerId ]['displayName']
                        )
                        response = {
                            'response': True,
                            'id': message['id'],
                            'ok': True,
                            'data': {}
                        }
                        await self._websocket.send(json.dumps(response))
                    elif message.get('method') == 'newDataConsumer':
                        await self.consumeData(
                            id=message['data']['id'],
                            dataProducerId=message['data']['dataProducerId'],
                            label=message['data']['label'],
                            protocol=message['data']['protocol'],
                            sctpStreamParameters=message['data']['sctpStreamParameters']
                        )
                        response = {
                            'response': True,
                            'id': message['data']['id'],
                            'ok': True,
                            'data': {}
                        }
                        await self._websocket.send(json.dumps(response))
                elif message.get('notification'):
                    print(message)
                    peer = message['data']
                    if message.get('method') == "newPeer" :
                        self.users[peer['id']] = {
                            "peerId" : peer['id'],
                            "displayName" : peer['displayName'],
                            "device" : peer['device'],
                            "cname" : "",
                            "comsumers" : []
                        }
                    elif message.get('method') == "peerClosed" :
                        print("peerClosed")
                        #if self.users[peer['id']]["displayName"] == "admin" :
                            #await self._recorder.stop()
                        #self.users.pop(peer['id'])

    # wait for answer ready        
    async def _wait_for(
        self, fut: Awaitable[T], timeout: Optional[float], **kwargs: Any
    ) -> T:
        try:
            return await asyncio.wait_for(fut, timeout=timeout, **kwargs)
        except asyncio.TimeoutError:
            raise Exception("Operation timed out")

    async def _send_request(self, request):
        self._answers[request['id']] = self._loop.create_future()
        await self._websocket.send(json.dumps(request))

    # Generates a random positive integer.
    def generateRandomNumber(self) -> int:
        return round(random() * 10000000)

    async def run(self):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        localhost_pem = pathlib.Path(__file__).with_name("fullchain.pem")
        ssl_context.load_verify_locations(localhost_pem)
        self._websocket = await websockets.connect(uri, subprotocols=['protoo'], ssl=ssl_context)
        if sys.version_info < (3, 7):
            task_run_recv_msg = asyncio.ensure_future(self.recv_msg_task())
        else:
            task_run_recv_msg = asyncio.create_task(self.recv_msg_task())
        self._tasks.append(task_run_recv_msg)

        await self.load()
        await self.createSendTransport()
        await self.createRecvTransport()
        await self.produce()

        await task_run_recv_msg
    
    async def load(self):
        # Init device
        self._device = Device(handlerFactory=AiortcHandler.createFactory(tracks=self._tracks))

        # Get Router RtpCapabilities
        reqId = self.generateRandomNumber()
        req = {
            'request': True,
            'id': reqId,
            'method': 'getRouterRtpCapabilities',
            'data': {}
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)

        print("!!!RouterRtpCapabilities:", ans)

        # Load Router RtpCapabilities
        await self._device.load(ans['data'])
    
    async def createSendTransport(self):
        if self._sendTransport != None:
            return
        # Send create sendTransport request
        reqId = self.generateRandomNumber()
        req = {
            'request': True,
            'id': reqId,
            'method': 'createWebRtcTransport',
            'data': {
                'forceTcp': False,
                'producing': True,
                'consuming': False,
                'sctpCapabilities': self._device.sctpCapabilities.dict()
            }
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)
        print("!!!!SendTransport:", ans)

        ans['data']['iceCandidates'][0]['ip'] = "172.16.1.221"

        # Create sendTransport
        self._sendTransport = self._device.createSendTransport(
            id=ans['data']['id'], 
            iceParameters=ans['data']['iceParameters'], 
            iceCandidates=ans['data']['iceCandidates'], 
            dtlsParameters=ans['data']['dtlsParameters'],
            sctpParameters=ans['data']['sctpParameters']
        )

        @self._sendTransport.on('connect')
        async def on_connect(dtlsParameters):
            reqId = self.generateRandomNumber()
            req = {
                "request":True,
                "id":reqId,
                "method":"connectWebRtcTransport",
                "data":{
                    "transportId": self._sendTransport.id,
                    "dtlsParameters": dtlsParameters.dict(exclude_none=True)
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
        
        @self._sendTransport.on('produce')
        async def on_produce(kind: str, rtpParameters, appData: dict):
            reqId = self.generateRandomNumber()
            req = {
                "id": reqId,
                'method': 'produce',
                'request': True,
                'data': {
                    'transportId': self._sendTransport.id,
                    'kind': kind,
                    'rtpParameters': rtpParameters.dict(exclude_none=True),
                    'appData': appData
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
            return ans['data']['id']

        @self._sendTransport.on('producedata')
        async def on_producedata(
            sctpStreamParameters: SctpStreamParameters,
            label: str,
            protocol: str,
            appData: dict
        ):
            
            reqId = self.generateRandomNumber()
            req = {
                "id": reqId,
                'method': 'produceData',
                'request': True,
                'data': {
                    'transportId': self._sendTransport.id,
                    'label': label,
                    'protocol': protocol,
                    'sctpStreamParameters': sctpStreamParameters.dict(exclude_none=True),
                    'appData': appData
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
            return ans['data']['id']

    async def produce(self):
        if self._sendTransport == None:
            await self.createSendTransport()
        
        # Join room
        reqId = self.generateRandomNumber()
        req = {
            "request":True,
            "id":reqId,
            "method":"join",
            "data":{
                "displayName":"pymediasoup",
                "device":{
                    "flag":"python",
                    "name":"python","version":"0.1.0"
                },
                "rtpCapabilities": self._device.rtpCapabilities.dict(exclude_none=True),
                "sctpCapabilities": self._device.sctpCapabilities.dict(exclude_none=True)
            }
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)

        print("!!!!join room ans:", ans)

        self.users = {}
        for peer in ans['data']['peers'] :
            self.users[peer['id']] = {
                    "peerId" : peer['id'],
                    "displayName" : peer['displayName'],
                    "device" : peer['device'],
                    "cname" : "",
                    "comsumers" : []
                }

        # produce
        """
        videoProducer: Producer = await self._sendTransport.produce(
            track=self._videoTrack,
            stopTracks=False,
            appData={}
        )
        self._producers.append(videoProducer)
        
        audioProducer: Producer = await self._sendTransport.produce(
            track=self._audioTrack,
            stopTracks=False,
            appData={}
        )
        self._producers.append(audioProducer)
        """

        # produce data
        await self.produceData()
        
    
    async def produceData(self):
        if self._sendTransport == None:
            await self.createSendTransport()

        """
        dataProducer: DataProducer = await self._sendTransport.produceData(
            ordered=False,
            maxPacketLifeTime=5555,
            label='chat',
            protocol='',
            appData={'info': "my-chat-DataProducer"}
        )
        self._producers.append(dataProducer)
        if not self._closed:
            await asyncio.sleep(1)
            dataProducer.send('hello')
        """
        

    
    async def createRecvTransport(self):
        if self._recvTransport != None:
            return
        # Send create recvTransport request
        reqId = self.generateRandomNumber()
        req = {
            'request': True,
            'id': reqId,
            'method': 'createWebRtcTransport',
            'data': {
                'forceTcp': False,
                'producing': False,
                'consuming': True,
                'sctpCapabilities': self._device.sctpCapabilities.dict()
            }
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)
        print("!!!!!RecvTransport:", ans)

        ans['data']['iceCandidates'][0]['ip'] = "172.16.1.221"

        # Create recvTransport
        self._recvTransport = self._device.createRecvTransport(
            id=ans['data']['id'], 
            iceParameters=ans['data']['iceParameters'], 
            iceCandidates=ans['data']['iceCandidates'], 
            dtlsParameters=ans['data']['dtlsParameters'],
            sctpParameters=ans['data']['sctpParameters']
        )

        @self._recvTransport.on('connect')
        async def on_connect(dtlsParameters):
            reqId = self.generateRandomNumber()
            req = {
                "request":True,
                "id":reqId,
                "method":"connectWebRtcTransport",
                "data":{
                    "transportId": self._recvTransport.id,
                    "dtlsParameters": dtlsParameters.dict(exclude_none=True)
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
            print("!!!!recv transport connected:", ans)
        
    async def consume(self, id, producerId, kind, rtpParameters, displayName):
        print(f"!!!!id:{id}, producerId:{producerId}, kind:{kind}, rtpParameters:{rtpParameters}")

        if self._recvTransport == None:
            await self.createRecvTransport()
        consumer: Consumer = await self._recvTransport.consume(
            id=id,
            producerId=producerId,
            kind=kind,
            rtpParameters=rtpParameters
        )
        self._consumers.append(consumer)
        #print(self._consumers)

        if kind == "video" and displayName==r"巡檢人員-施文凱(手機)" :
            if self._recorder != None :
                self.comsumer_changed = True
                self._new_loopback_video_track = consumer.track
            else :
                self._loopback_video_track = consumer.track   

        elif kind == "audio" and displayName==r"巡檢人員-施文凱(手機)" :
            if self._recorder != None :
                self.comsumer_changed = True
                self._new_loopback_audio_track = consumer.track
            else :      
                self._loopback_audio_track = consumer.track
        else :
            blackhole = MediaBlackhole()
            blackhole.addTrack(consumer.track)
            blackhole.start()
            self._blackholes.append(blackhole)        

        if self._loopback_video_track != None and self._loopback_audio_track != None and self._recorder == None :

            self._consume_task = asyncio.ensure_future(comsume_remote_track(self._loopback_video_track))



        if self._new_loopback_video_track != None and self._new_loopback_audio_track != None and self.comsumer_changed == True :
            #await self._recorder.stop()

            self._loopback_video_track = self._new_loopback_video_track
            self._loopback_video_track = self._new_loopback_video_track

            self._new_loopback_audio_track = None
            self._new_loopback_video_track = None

            self._consume_task.cancel()

            self._consume_task = None

            self._consume_task = asyncio.ensure_future(comsume_remote_track(self._loopback_video_track))

            #self._videoTrack.stopComsuming()

            #self._videoTrack.startComsuming(self._loopback_video_track)

            

            #self._recorder = MediaRecorder(f"rtsp://127.0.0.1:8554/mystream", format='rtsp', options={"mux_delay":'0.1', 'rtsp_transport': 'udp'})

            #self._loopback_audio_track = self._new_loopback_audio_track
            #self._loopback_video_track = self._new_loopback_video_track

            #self._new_loopback_audio_track = None
            #self._new_loopback_video_track = None

            #self._recorder.addTrack(self._loopback_video_track)
            #self._recorder.addTrack(self._loopback_audio_track)

            #self.comsumer_changed == False

            #await self._recorder.start()
            

            


        """
        print(type(consumer.track))

        if kind == "video" and self._loopback_video_track == None :
            self._loopback_video_track = consumer.track

        if kind == "audio" and self._loopback_audio_track == None :
            self._loopback_audio_track = consumer.track

        if self._loopback_video_track != None and self._loopback_audio_track != None :
            #self.media_relay = MediaRelay()
            
            #self.audio_relay_track = self.media_relay.subscribe(self._loopback_audio_track)
            try :
                
                self._videoTrack.startComsuming(self._loopback_video_track)
            

                #frame = await self.audio_relay_track.recv()
                #a_frame = await self.video_relay_track.recv()
        """
                #print("!!!!!!!!start produdeing video??")
                
                #videoProducer: Producer = await self._sendTransport.produce(
                #    track=self._videoTrack,
                #    stopTracks=False,
                #    appData={}
                #)
                #self._producers.append(videoProducer)
                
                #audioProducer: Producer = await self._sendTransport.produce(
                #    track=self.audio_relay_track,
                #    stopTracks=False,
                #    appData={}
                #)
                #self._producers.append(audioProducer)
        """


            except Exception as e :
                print(e)
        """

            

    async def consumeData(self, id, dataProducerId, sctpStreamParameters, label=None, protocol=None, appData={}):
        pass
        dataConsumer: DataConsumer = await self._recvTransport.consumeData(
            id=id,
            dataProducerId=dataProducerId,
            sctpStreamParameters=sctpStreamParameters,
            label=label,
            protocol=protocol,
            appData=appData
        )
        self._consumers.append(dataConsumer)
        @dataConsumer.on('message')
        def on_message(message):
            print(f'DataChannel {label}-{protocol}: {message}')

    
    async def close(self):
        for consumer in self._consumers:
            await consumer.close()
        for producer in self._producers:
            await producer.close()
        for task in self._tasks:
            task.cancel()
        if self._sendTransport:
            await self._sendTransport.close()
        if self._recvTransport:
            await self._recvTransport.close()
        for key, recorder in self._recorder_dict.items():
            await recorder.stop()

def sender():
    time.sleep(3)
    options = {"multiclient_mode": True}
    global upload_image

    gear_server = NetGear(
    address="127.0.0.1",
    port=(5567,5577,5588),
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)   
    gear_server.send(cv2.imread('black.jpg'))
    while True:
        try:
            img=upload_image

            if len(upload_image)>0:
                #print(img)
                gear_server.send(img)
            else:
                gear_server.send(cv2.imread('black.jpg'))
        except Exception as e:
            gear_server.send(cv2.imread('black.jpg'))
            print(e)
            pass
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyMediaSoup")
    parser.add_argument("room", nargs="?")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument("--record-to", help="Write received media to a file.")
    args = parser.parse_args()

    room = "admin-console"
    t = threading.Thread(target = sender)
    # 執行該子執行緒
    t.start()
    if not args.room:
        args.room = secrets.token_urlsafe(8).lower()
    peerId = secrets.token_urlsafe(8).lower()

    uri = f'wss://127.0.0.1:4443/?roomId={room}&peerId={peerId}'

    print(uri)

    if args.play_from:
        player = MediaPlayer(args.play_from)
    else:
        player = None
    
    # create media sink
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    # run event loop
    loop = asyncio.get_event_loop()
    try:
        demo = Demo(uri=uri, player=player, recorder=recorder, loop=loop)
        loop.run_until_complete(demo.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(demo.close())







