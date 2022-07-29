from socket import *
import json

import cv2
import time
import os
import math
import ctypes
import signal
import sys
import numpy as np
import threading

from multiprocessing import Process ,Queue

from copy import deepcopy

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from srs import *

from PyQt5.QtCore import QTimer



cam0_lock = threading.Lock()
cam1_lock = threading.Lock()

pi1CapLock = threading.Lock()

#cam0_delayFrames = 0
#cam1_delayFrames = 0

pih = 600
piw = 600

dw = 320
dh = 320

blackFrame = np.zeros((dw,dh),dtype=np.uint32)

def transToConsole(x, y, w, h):
    return int(x * dw / piw), int(y * dh / pih), int(w * dw / piw), int(h * dh / pih)

def transToPi(x, y, w, h):
    return int(x * piw / dw), int(y *  pih/ dh), int(w *  piw/dw ), int(h *  pih/dh )
def scaleToPi(val):
    return int(val * piw / dw)


class Pi1CamThread(threading.Thread):
    def __init__(self, cam0_path, height, width):
        super(Pi1CamThread, self).__init__()
        
        self.cam0_path = cam0_path

        self.img_width = width
        self.img_height = height

        self.cam0_available = False

        self.cam0_frame = None

        self.running = False
        

    def run(self):
        self.running = True
        self.cam0 = cv2.VideoCapture(self.cam0_path)
        self.cam0.read()
        while self.running:
            ret, frame = self.cam0.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                pi1CapLock.acquire()
                self.cam0_frame = frame
                #self.cam0_frames.append(frame)
                pi1CapLock.release()
                self.cam0_available = True
            else:
                print("picam0 error")
                time.sleep(0.5)
                self.cam0 = cv2.VideoCapture(self.cam0_path)


        self.cam0.release()

    def get_frame(self):
        cnt = 0
        while not self.cam0_available:
            time.sleep(0.01)
            cnt = cnt + 1
            if cnt > 5:
                if self.cam0_frame is not None:
                    return (self.cam0_frame)
        self.cam0_available = False
        #return deepcopy(self.cam0_frame)
        return (self.cam0_frame)
        #return (self.cam0_frames.pop(0))

    def exit(self):
        self.running = False
    
    
class Pi2CamThread(threading.Thread):
    def __init__(self, cam0_path, height, width):
        super(Pi2CamThread, self).__init__()
        
        self.cam0_path = cam0_path

        self.img_width = width
        self.img_height = height

        self.cam0_available = False

        self.cam0_frame = None

        self.running = False
        

    def run(self):
        self.running = True
        self.cam0 = cv2.VideoCapture(self.cam0_path)
        self.cam0.read()
        while self.running:
            ret, frame = self.cam0.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                pi1CapLock.acquire()
                self.cam0_frame = frame
                #self.cam0_frames.append(frame)
                pi1CapLock.release()
                self.cam0_available = True
            else:
                print("picam1 error")
                time.sleep(0.5)
                self.cam0 = cv2.VideoCapture(self.cam0_path)


        self.cam0.release()

    def get_frame(self):
        cnt = 0
        while not self.cam0_available:
            time.sleep(0.01)
            cnt = cnt + 1
            if cnt > 5:
                if self.cam0_frame is not None:
                    return (self.cam0_frame)
                #return blackFrame
        self.cam0_available = False
        #return deepcopy(self.cam0_frame)
        return (self.cam0_frame)
        #return (self.cam0_frames.pop(0))

    def exit(self):
        self.running = False    



class Camera1Thread(threading.Thread):
    def __init__(self, cam0_path, height, width):
        super(Camera1Thread, self).__init__()

        self.cam0_path = cam0_path

        self.cam0 = cv2.VideoCapture(self.cam0_path)

        self.img_width = width
        self.img_height = height

        self.cam0_available = False

        self.cam0_frame = None

        self.running = False
        
        self.cam0_frames = []

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cam0.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                cam0_lock.acquire()
                self.cam0_frame = frame
                #self.cam0_frames.append(frame)
                cam0_lock.release()
                self.cam0_available = True
            else:
                print("cam0 error")
                time.sleep(1)
                self.cam0 = cv2.VideoCapture(self.cam0_path)


        self.cam0.release()

    def get_frame(self):
        cnt = 0
        while not self.cam0_available:
            time.sleep(0.02)
            cnt = cnt + 1
            if cnt > 100:
                return None
        self.cam0_available = False
        #return deepcopy(self.cam0_frame)
        return (self.cam0_frame)
        #return (self.cam0_frames.pop(0))

    def exit(self):
        self.running = False

class Camera2Thread(threading.Thread):
    def __init__(self, cam1_path, height, width):
        super(Camera2Thread, self).__init__()

        self.cam1_path = cam1_path

        self.cam1 = cv2.VideoCapture(self.cam1_path)

        self.img_width = width
        self.img_height = height

        self.cam1_available = False

        self.cam1_frame = None

        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cam1.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                cam1_lock.acquire()
                self.cam1_frame = frame
                cam1_lock.release()
                self.cam1_available = True
            else:
                print("cam1 error")
                time.sleep(1)
                self.cam1 = cv2.VideoCapture(self.cam1_path)


        self.cam1.release()

    def get_frame(self):
        cnt = 0
        while not self.cam1_available:
            time.sleep(0.02)
            cnt = cnt + 1
            if cnt > 100:
                return None
        self.cam1_available = False
        #return deepcopy(self.cam1_frame)
        return (self.cam1_frame)

    def exit(self):
        self.running = False



class ServerThread(threading.Thread):
    def __init__(self):
        super(ServerThread, self).__init__()
        self.s = socket(AF_INET, SOCK_DGRAM)
        
        self.s.bind(('0.0.0.0', 14400))
        
        self.running = False

        self.cam0_dat = None
        self.cam1_dat = None
        
        self.cam0_addr = None
        self.cam1_addr = None

    def run(self):
        self.running = True
        while self.running:
            data, address = self.s.recvfrom(1024)

            jsdat = json.loads(data.decode('UTF-8'))
            # print('Accept message:' + data.decode('ascii'))
            if jsdat['id'] == 0:
                self.cam0_dat = jsdat
                self.cam0_addr = address
            if jsdat['id'] == 1:
                self.cam1_dat = jsdat
                self.cam1_addr = address
                
            # print(jsdat['loc'])

        # Reply = input('Send message:')
        # s.sendto(Reply.encode('ascii'),address)
        self.s.close()

    def get_cam_dat(self, id):

        if id == 0:
            return self.cam0_dat
        if id == 1:
            return self.cam1_dat

    def send_cam_dat(self, id, dat):
        if id == 0:
            if self.cam0_addr is not None:
                self.s.sendto(dat, self.cam0_addr)
        if id == 1:
            if self.cam1_addr is not None:
                self.s.sendto(dat, self.cam1_addr)        

    def exit(self):
        self.running = False
        pass


class WindowM(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(WindowM, self).__init__(parent)
        self.setupUi(self)
        
        self.cap0_timer = QTimer()
        self.cap0_timer.timeout.connect(self.cap0_timer_cb)
        self.cap0_timer.start(50)
        
        self.cap1_timer = QTimer()
        self.cap1_timer.timeout.connect(self.cap1_timer_cb)
        self.cap1_timer.start(50)        
        
        self.calc_timer = QTimer()
        self.calc_timer.timeout.connect(self.calc_timer_cb)
        self.calc_timer.start(50)    

        
        scene0 = QGraphicsScene(self)
        self.pixmap_item0 = QGraphicsPixmapItem()
        scene0.addItem(self.pixmap_item0)
        self.graphicsView.setScene(scene0)
        
        scene1 = QGraphicsScene(self)
        self.pixmap_item1 = QGraphicsPixmapItem()
        scene1.addItem(self.pixmap_item1)
        self.graphicsView_2.setScene(scene1)       
        
        self.thSyn1 = 0
        self.thSyn2 = 0
        
        self.loc0_list = []
        self.loc1_list = []
        
        self.cam0_info_list = []
        self.cam1_info_list = []
        
        self.cam0DetSizeChanging = False
        self.cam1DetSizeChanging = False
        
        #===================================== CAM0 =====================================
        self.dial.valueChanged.connect(self.dial_tri_cb)
        self.dial_2.valueChanged.connect(self.dial_2_tri_cb)
        self.dial_3.valueChanged.connect(self.dial_3_tri_cb)
        self.dial_4.valueChanged.connect(self.dial_4_tri_cb)
        self.dial_5.valueChanged.connect(self.dial_5_tri_cb)
        self.dial_6.valueChanged.connect(self.dial_6_tri_cb)
        self.dial_7.valueChanged.connect(self.dial_7_tri_cb)
        self.dial_8.valueChanged.connect(self.dial_8_tri_cb)
        self.dial_9.valueChanged.connect(self.dial_9_tri_cb)
        self.dial_10.valueChanged.connect(self.dial_10_tri_cb)
        
        self.dial_3.sliderPressed.connect(self.cam0DetSizeChangStart)
        self.dial_4.sliderPressed.connect(self.cam0DetSizeChangStart)
        self.dial_5.sliderPressed.connect(self.cam0DetSizeChangStart)
        self.dial_6.sliderPressed.connect(self.cam0DetSizeChangStart)        
        self.dial_7.sliderPressed.connect(self.cam0DetSizeChangStart)
        self.dial_8.sliderPressed.connect(self.cam0DetSizeChangStart)
        self.dial_9.sliderPressed.connect(self.cam0DetSizeChangStart)
        self.dial_10.sliderPressed.connect(self.cam0DetSizeChangStart)
         
        self.dial_3.sliderReleased.connect(self.cam0DetSizeChangFinish)
        self.dial_4.sliderReleased.connect(self.cam0DetSizeChangFinish)
        self.dial_5.sliderReleased.connect(self.cam0DetSizeChangFinish)
        self.dial_6.sliderReleased.connect(self.cam0DetSizeChangFinish)        
        self.dial_7.sliderReleased.connect(self.cam0DetSizeChangFinish)
        self.dial_8.sliderReleased.connect(self.cam0DetSizeChangFinish)
        self.dial_9.sliderReleased.connect(self.cam0DetSizeChangFinish)
        self.dial_10.sliderReleased.connect(self.cam0DetSizeChangFinish)
        
        self.radioButton.toggled.connect(self.radb_tog)  
        self.radioButton_4.toggled.connect(self.radb_4_tog)  
        self.radioButton_9.toggled.connect(self.radb_9_tog)  
        self.radioButton_10.toggled.connect(self.radb_10_tog)  
        #===============================================================================


        self.dial_11.valueChanged.connect(self.dial_11_tri_cb)
        self.dial_12.valueChanged.connect(self.dial_12_tri_cb)
        self.dial_13.valueChanged.connect(self.dial_13_tri_cb)
        self.dial_14.valueChanged.connect(self.dial_14_tri_cb)
        self.dial_15.valueChanged.connect(self.dial_15_tri_cb)
        self.dial_16.valueChanged.connect(self.dial_16_tri_cb)
        self.dial_18.valueChanged.connect(self.dial_18_tri_cb)
        self.dial_20.valueChanged.connect(self.dial_20_tri_cb)
        self.dial_17.valueChanged.connect(self.dial_17_tri_cb)
        self.dial_19.valueChanged.connect(self.dial_19_tri_cb)
        
        self.dial_13.sliderPressed.connect(self.cam1DetSizeChangStart)
        self.dial_14.sliderPressed.connect(self.cam1DetSizeChangStart)
        self.dial_15.sliderPressed.connect(self.cam1DetSizeChangStart)
        self.dial_16.sliderPressed.connect(self.cam1DetSizeChangStart)        
        self.dial_18.sliderPressed.connect(self.cam1DetSizeChangStart)
        self.dial_20.sliderPressed.connect(self.cam1DetSizeChangStart)
        self.dial_17.sliderPressed.connect(self.cam1DetSizeChangStart)
        self.dial_19.sliderPressed.connect(self.cam1DetSizeChangStart)
         
        self.dial_13.sliderReleased.connect(self.cam1DetSizeChangFinish)
        self.dial_14.sliderReleased.connect(self.cam1DetSizeChangFinish)
        self.dial_15.sliderReleased.connect(self.cam1DetSizeChangFinish)
        self.dial_16.sliderReleased.connect(self.cam1DetSizeChangFinish)        
        self.dial_18.sliderReleased.connect(self.cam1DetSizeChangFinish)
        self.dial_20.sliderReleased.connect(self.cam1DetSizeChangFinish)
        self.dial_17.sliderReleased.connect(self.cam1DetSizeChangFinish)
        self.dial_19.sliderReleased.connect(self.cam1DetSizeChangFinish)
        
        self.radioButton_5.toggled.connect(self.radb_5_tog)  
        self.radioButton_8.toggled.connect(self.radb_8_tog)  
        self.radioButton_11.toggled.connect(self.radb_11_tog)  
        self.radioButton_12.toggled.connect(self.radb_12_tog)          
        
        
        self.pushButton_2.clicked.connect(self.button_2_click)
        self.pushButton.clicked.connect(self.button_click)
    
    
        self.runTimer = QTimer()
        self.runTimer.timeout.connect(self.runTimer_cb)
        
        
        self.countDownWait = 0
        self.lenList = []
        self.argList = []
        
        self.startRun = False
        
        self.resultLen = 0
        self.resultArg = 0
        
        self.lengFin = False
        self.argFin = False
    
        self.lenListLastN = []
        self.argListLastN = []
        
        self.lastGetLen = 0
        self.lastGetArg = 0
        
        self.pastTime = 0
    
    def runTimer_cb(self):
        
        self.pastTime = self.pastTime + 1
        self.lcdNumber_5.setProperty("value", self.pastTime)
        
        
        if(self.countDownWait == 0):
            self.label_27.setText("Running...")
            self.startRun = True
            
            if(len(self.lenList) > 10):
                self.lenListLastN = self.lenList[::-1][0:10]
                self.lenListLastN.remove(max(self.lenListLastN))
                self.lenListLastN.remove(min(self.lenListLastN))
                
                #print(np.mean(self.lenListLastN))
                if(np.max(self.lenListLastN) - np.min(self.lenListLastN) < 4 and not self.lengFin):
                    self.resultLen = float(np.mean(self.lenListLastN))
                    self.lcdNumber.setProperty("value", self.resultLen)
                    self.lengFin = True
                
                
            if(len(self.argList) > 10):
                
                self.argListLastN = self.argList[::-1][0:10]  
                self.argListLastN.remove(max(self.argListLastN))
                self.argListLastN.remove(min(self.argListLastN))
                #print(np.mean(self.argListLastN))
                if(np.max(self.argListLastN) - np.min(self.argListLastN) < 6 and not self.argFin):
                    
                    self.resultArg = float(np.mean(self.argListLastN))
                    self.lcdNumber_2.setProperty("value", self.resultArg)
                    self.argFin = True
            
            if(self.pastTime >= 27):
                print("TimeOut")
                if not self.lengFin:
                    self.lengFin = True
                    self.resultLen = float(np.mean(self.lenList[::-1][0:3]))
                    self.lcdNumber.setProperty("value", self.resultLen)
                
                if not self.argFin:
                    self.argFin = True
                    self.resultArg = float(np.mean(self.argList[::-1][0:3]))
                    self.lcdNumber_2.setProperty("value", self.resultArg)
            
            if (self.lengFin and self.argFin):
                self.startRun = False
                self.runTimer.stop()
                self.label_27.setText("Finished.")
                
                
                
                
                
            
        else:
            self.countDownWait = self.countDownWait - 1
            self.label_27.setText("Starting...")
            jsdat = json.dumps(
                    {
                        'action':'reset',
                    }
                )
            server.send_cam_dat(0, jsdat.encode('UTF-8'))
            server.send_cam_dat(1, jsdat.encode('UTF-8'))    

            
    
    def button_click(self):
        self.label_27.setText("Starting...")
        self.runTimer.stop()
        self.pastTime = 0
        self.lcdNumber.setProperty("value", 0)
        self.lcdNumber_2.setProperty("value", 0)
        self.startRun = False
        self.resultLen = 0
        self.resultArg = 0
        
        self.lenListLastN = []
        self.argListLastN = []
        self.lenList = []
        self.argList = []
        
        self.lengFin = False
        self.argFin = False
        
        #self.countDownWait = 5
        self.countDownWait = 1
        self.runTimer.start(1000)  
    
    
    def button_2_click(self):
        self.pastTime = 0
        self.label_27.setText("STOP")
        self.runTimer.stop()
        self.startRun = False
    
        self.lcdNumber.setProperty("value", 0)
        self.lcdNumber_2.setProperty("value", 0)
        
        self.lenList = []
        self.argList = []
        
        jsdat = json.dumps(
                {
                    'action':'reset',
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))
        server.send_cam_dat(1, jsdat.encode('UTF-8'))

    
    def cam0DetSizeChangStart(self):
        self.cam0DetSizeChanging = True
    def cam0DetSizeChangFinish(self):
        self.cam0DetSizeChanging = False        
        
    def dial_tri_cb(self, int):
        self.plainTextEdit.setPlainText(str(self.dial.value()))
        jsdat = json.dumps(
                {
                    'action':'setDiffThresh',
                    'val':self.dial.value()
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))
        
    def dial_2_tri_cb(self, int):
        if(self.dial_2.value() % 2 == 0):
            self.dial_2.setValue(self.dial_2.value() + 1)
        self.plainTextEdit_2.setPlainText(str(self.dial_2.value()))
        jsdat = json.dumps(
                {
                    'action':'setKernelSize',
                    'val':self.dial_2.value()
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))        
        
    def dial_3_tri_cb(self, int):
        self.plainTextEdit_3.setPlainText(str(self.dial_3.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinXoffs',
                    'val':scaleToPi(self.dial_3.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))          

    def dial_4_tri_cb(self, int):
        self.plainTextEdit_4.setPlainText(str(self.dial_4.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinYoffs',
                    'val':scaleToPi(self.dial_4.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        
        
    def dial_5_tri_cb(self, int):
        self.plainTextEdit_5.setPlainText(str(self.dial_5.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinW',
                    'val':scaleToPi(self.dial_5.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8')) 
        
    def dial_6_tri_cb(self, int):
        self.plainTextEdit_6.setPlainText(str(self.dial_6.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinH',
                    'val':scaleToPi(self.dial_6.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        
    def dial_7_tri_cb(self, int):
        self.plainTextEdit_7.setPlainText(str(self.dial_7.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxXoffs',
                    'val':scaleToPi(self.dial_7.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        
        
    def dial_8_tri_cb(self, int):
        self.plainTextEdit_10.setPlainText(str(self.dial_8.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxYoffs',
                    'val':scaleToPi(self.dial_8.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        
        
    def dial_9_tri_cb(self, int):
        self.plainTextEdit_8.setPlainText(str(self.dial_9.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxW',
                    'val':scaleToPi(self.dial_9.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        
        
    def dial_10_tri_cb(self, int):
        self.plainTextEdit_9.setPlainText(str(self.dial_10.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxH',
                    'val':scaleToPi(self.dial_10.value())
                }
            )
        server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        

    def radb_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':-1
                }
            )
            server.send_cam_dat(0, jsdat.encode('UTF-8'))          
    def radb_4_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':0
                }
            )
            server.send_cam_dat(0, jsdat.encode('UTF-8'))  
    def radb_9_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':1
                }
            )
            server.send_cam_dat(0, jsdat.encode('UTF-8'))  
    def radb_10_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':2
                }
            )
            server.send_cam_dat(0, jsdat.encode('UTF-8'))  
        
        
        
        
    #============================================================================
    
    
    

    def cam1DetSizeChangStart(self):
        self.cam1DetSizeChanging = True
    def cam1DetSizeChangFinish(self):
        self.cam1DetSizeChanging = False        
        
    def dial_11_tri_cb(self, int):
        self.plainTextEdit_11.setPlainText(str(self.dial_11.value()))
        jsdat = json.dumps(
                {
                    'action':'setDiffThresh',
                    'val':self.dial_11.value()
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))
        
    def dial_12_tri_cb(self, int):
        if(self.dial_12.value() % 2 == 0):
            self.dial_12.setValue(self.dial_12.value() + 1)
        self.plainTextEdit_12.setPlainText(str(self.dial_12.value()))
        jsdat = json.dumps(
                {
                    'action':'setKernelSize',
                    'val':self.dial_12.value()
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))        
        
    def dial_13_tri_cb(self, int):
        self.plainTextEdit_13.setPlainText(str(self.dial_13.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinXoffs',
                    'val':scaleToPi(self.dial_13.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))          

    def dial_14_tri_cb(self, int):
        self.plainTextEdit_14.setPlainText(str(self.dial_14.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinYoffs',
                    'val':scaleToPi(self.dial_14.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))  
        
        
    def dial_15_tri_cb(self, int):
        self.plainTextEdit_15.setPlainText(str(self.dial_15.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinW',
                    'val':scaleToPi(self.dial_15.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8')) 
        
    def dial_16_tri_cb(self, int):
        self.plainTextEdit_16.setPlainText(str(self.dial_16.value()))
        jsdat = json.dumps(
                {
                    'action':'setMinH',
                    'val':scaleToPi(self.dial_16.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))  
        
    def dial_17_tri_cb(self, int):
        self.plainTextEdit_19.setPlainText(str(self.dial_17.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxXoffs',
                    'val':scaleToPi(self.dial_17.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))  
        
        
    def dial_18_tri_cb(self, int):
        self.plainTextEdit_18.setPlainText(str(self.dial_18.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxYoffs',
                    'val':scaleToPi(self.dial_18.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))  
        
        
    def dial_19_tri_cb(self, int):
        self.plainTextEdit_20.setPlainText(str(self.dial_19.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxW',
                    'val':scaleToPi(self.dial_19.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))  
        
        
    def dial_20_tri_cb(self, int):
        self.plainTextEdit_17.setPlainText(str(self.dial_20.value()))
        jsdat = json.dumps(
                {
                    'action':'setMaxH',
                    'val':scaleToPi(self.dial_20.value())
                }
            )
        server.send_cam_dat(1, jsdat.encode('UTF-8'))  
        

    def radb_5_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':-1
                }
            )
            server.send_cam_dat(1, jsdat.encode('UTF-8'))          
    def radb_8_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':0
                }
            )
            server.send_cam_dat(1, jsdat.encode('UTF-8'))  
    def radb_11_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':1
                }
            )
            server.send_cam_dat(1, jsdat.encode('UTF-8'))  
    def radb_12_tog(self, status):
        if status:
            jsdat = json.dumps(
                {
                    'action':'chPushMode',
                    'val':2
                }
            )
            server.send_cam_dat(1, jsdat.encode('UTF-8'))      
        
        
        
        
        
        
        
    def cap0_timer_cb(self):
    
        (x, y, w, h) = (0, 0, 0, 0)
        global blackFrame
        
        if self.radioButton_3.isChecked() or self.radioButton_2.isChecked():
            frame = camera0.get_frame()
        elif self.radioButton_4.isChecked() or self.radioButton_9.isChecked() or self.radioButton_10.isChecked():
            frame = pi1cap.get_frame()
        else:
            frame = blackFrame
        
        if type(frame) != type(None):
            cam0_dat = server.get_cam_dat(0)
            if cam0_dat is not None:
                self.loc0_list.append(cam0_dat['loc'])
                self.cam0_info_list.append(cam0_dat)
                if len(self.loc0_list) > self.thSyn1:
                    x, y, w, h = self.loc0_list.pop(0)
                    x, y, w, h = transToConsole(x, y, w, h)
                    if self.radioButton_3.isChecked() :
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3, 1)        
                    
                    
            if self.cam0DetSizeChanging:
                sp = (80,160)
                if((sp[0]) + self.dial_3.value() < 320 and (sp[1]) + self.dial_4.value() < 320):
                    ep = ((sp[0]) + self.dial_3.value() ,(sp[1]) + self.dial_4.value() )
                else:
                    ep = (320, 320)
                cv2.arrowedLine(frame, sp, (ep[0], sp[1]), (0, 255 ,0), 2)
                cv2.arrowedLine(frame, sp, (sp[0], ep[1]), (0, 255 ,0), 2)
                
                sp = (120,160)
                if((sp[0]) + self.dial_9.value() < 320 and (sp[1]) + self.dial_7.value() < 320):
                    ep = ((sp[0]) + self.dial_9.value() ,(sp[1]) + self.dial_7.value() )
                else:
                    ep = (320, 320)
                cv2.arrowedLine(frame, sp, (ep[0], sp[1]), (255, 0 ,0), 2)
                cv2.arrowedLine(frame, sp, (sp[0], ep[1]), (255, 0 ,0), 2)
                
                sp = (x,y)
                if((sp[0]) + self.dial_10.value() < 320 and (sp[1]) + self.dial_8.value() < 320):
                    ep = ((sp[0]) + self.dial_10.value() ,(sp[1]) + self.dial_8.value() )
                else:
                    ep = (320, 320)
                cv2.rectangle(frame, sp, ep, (255, 0, 0), 1, 1)  
                
                
                sp = (x,y)
                if((sp[0]) + self.dial_5.value() < 320 and (sp[1]) + self.dial_6.value() < 320):
                    ep = ((sp[0]) + self.dial_5.value() ,(sp[1]) + self.dial_6.value() )
                else:
                    ep = (320, 320)
                cv2.rectangle(frame, sp, ep, (0, 255, 0), 1, 1)      
                
                
            ch = ctypes.c_char.from_buffer(frame, 0)
            rcount = ctypes.c_long.from_address(id(ch)).value
        
            qframe = QImage(ch, dw, dh, QImage.Format_RGB888)
            
            ctypes.c_long.from_address(id(ch)).value = rcount
            
            qpframe = QPixmap.fromImage(qframe)
            self.pixmap_item0.setPixmap(qpframe)

       
    def cap1_timer_cb(self):            
            
        (x, y, w, h) = (0, 0, 0, 0)
        global blackFrame
        
        if self.radioButton_6.isChecked() or self.radioButton_7.isChecked():
            frame = camera1.get_frame()
        elif self.radioButton_8.isChecked() or self.radioButton_11.isChecked() or self.radioButton_12.isChecked():
            frame = pi2cap.get_frame()
        else:
            frame = blackFrame
        
        if type(frame) != type(None):
            cam1_dat = server.get_cam_dat(1)
            if cam1_dat is not None:
                self.loc1_list.append(cam1_dat['loc'])
                self.cam1_info_list.append(cam1_dat)
                if len(self.loc1_list) > self.thSyn2:
                    x, y, w, h = self.loc1_list.pop(0)
                    x, y, w, h = transToConsole(x, y, w, h)
                    if self.radioButton_7.isChecked() :
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3, 1)  
                            
            if self.cam1DetSizeChanging:
                sp = (80,160)
                if((sp[0]) + self.dial_13.value() < 320 and (sp[1]) + self.dial_14.value() < 320):
                    ep = ((sp[0]) + self.dial_13.value() ,(sp[1]) + self.dial_14.value() )
                else:
                    ep = (320, 320)
                cv2.arrowedLine(frame, sp, (ep[0], sp[1]), (0, 255 ,0), 2)
                cv2.arrowedLine(frame, sp, (sp[0], ep[1]), (0, 255 ,0), 2)
                
                sp = (120,160)
                if((sp[0]) + self.dial_17.value() < 320 and (sp[1]) + self.dial_18.value() < 320):
                    ep = ((sp[0]) + self.dial_17.value() ,(sp[1]) + self.dial_18.value() )
                else:
                    ep = (320, 320)
                cv2.arrowedLine(frame, sp, (ep[0], sp[1]), (255, 0 ,0), 2)
                cv2.arrowedLine(frame, sp, (sp[0], ep[1]), (255, 0 ,0), 2)
                
                sp = (x,y)
                if((sp[0]) + self.dial_19.value() < 320 and (sp[1]) + self.dial_20.value() < 320):
                    ep = ((sp[0]) + self.dial_19.value() ,(sp[1]) + self.dial_20.value() )
                else:
                    ep = (320, 320)
                cv2.rectangle(frame, sp, ep, (255, 0, 0), 1, 1)  
                
                
                sp = (x,y)
                if((sp[0]) + self.dial_15.value() < 320 and (sp[1]) + self.dial_16.value() < 320):
                    ep = ((sp[0]) + self.dial_15.value() ,(sp[1]) + self.dial_16.value() )
                else:
                    ep = (320, 320)
                cv2.rectangle(frame, sp, ep, (0, 255, 0), 1, 1)                 
    
            
            ch = ctypes.c_char.from_buffer(frame, 0)
            rcount = ctypes.c_long.from_address(id(ch)).value
            
            qframe = QImage(ch, dw, dh, QImage.Format_RGB888)
            ctypes.c_long.from_address(id(ch)).value = rcount
            
            qpframe = QPixmap.fromImage(qframe)
            self.pixmap_item1.setPixmap(qpframe) 
            
            
            
            
    
    def calc_timer_cb(self):
        if len(self.cam0_info_list) > self.thSyn1 and len(self.cam1_info_list) > self.thSyn2:
            cam0_dat = self.cam0_info_list.pop(0)
            cam1_dat = self.cam1_info_list.pop(0)
            if(abs(cam0_dat['line_length'] - cam1_dat['line_length']) < 10):
                length = (cam0_dat['line_length'] + cam1_dat['line_length'])/2
                      
                
                if self.startRun:
                    if length != self.lastGetLen:
                        self.lenList.append(length)
                        print("len list len:%d :" % (len(self.lenList)), np.round(self.lenList,2) )
                        self.lastGetLen = length
        
                self.lcdNumber_4.setProperty("value", length)
            if(cam0_dat['x_max_len'] > 0 and cam1_dat['x_max_len'] > 0):
                arg = math.degrees(math.atan(cam1_dat['x_max_len']/cam0_dat['x_max_len']))  
                
                if self.startRun:
                    if arg != self.lastGetArg:
                        self.argList.append(arg)
                        print("arg list len:%d :" % (len(self.argList)), np.round(self.argList,2))
                        self.lastGetArg = arg
                
                self.lcdNumber_3.setProperty("value", arg )
                
            



#/h264/ch1/sub/av_stream
pi1cap = Pi1CamThread("rtmp://192.168.5.122/live/1", dw, dh)  
pi2cap = Pi2CamThread("rtmp://192.168.5.122/live/2", dw, dh)  

camera0 = Camera1Thread("rtsp://admin:a1234567@192.168.5.120", dw, dh)
#camera0 = Camera1Thread("rtmp://192.168.5.122/hls/1", dw, dh)
camera1 = Camera2Thread("rtsp://admin:a1234567@192.168.5.121", dw, dh)                      
                      
                    
                     
server = ServerThread()


if __name__ == '__main__':
    camera0.start()
    camera1.start()
    
    pi1cap.start()
    pi2cap.start()
    
    server.start()
    
    app = QApplication(sys.argv)
    ex = WindowM()
    ex.show()
    ret = app.exec_()
    
    server.exit()
    camera0.exit()
    camera1.exit()
    pi1cap.exit()
    pi2cap.exit()
    
    
    sys.exit(ret)

    
    
