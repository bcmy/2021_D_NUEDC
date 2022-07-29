import cv2
import time
import math
import sys
import numpy as np
import threading
import json

from main_nanodet import my_nanodet
from copy import deepcopy

from socket import *




#capFrameMethod = "CSRT"
#capFrameMethod = "f_frame_diff"
capFrameMethod = "frame_diff"

#frame_diff
#nanodet
firstCapMethod = "nanodet"
ReCapMethod = "nanodet"

isFirstCap = True

autoSwitch = 1

g = 9.806
line_cut_off = 6
line_cut_off2 = 5

thread_lock = threading.Lock()
thread_exit = False

start_time = time.time()
time_100ms = time.time()

fps_cnt = 0

last_past_loc = False
lastCrossToRightTime =  time.time()
lastCrossToLeftTime =  time.time()
CrossToRightPeriod = 0
CrossToLeftPeriod = 0
CrossPeriod = 0

detExceptionCnt = 0

if(capFrameMethod == "f_frame_diff" or capFrameMethod == "frame_diff"):
    detLine = 600 / 2
else:
    detLine = 240 / 2

runingTime = 0

periodList = [0] * 10

line_length = 0

last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)

kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差

class clientThread(threading.Thread):
    def __init__(self, ip, port):
        super(clientThread, self).__init__()
        
        self.running = False
        self.address = (ip, port)
        self.s = socket(AF_INET, SOCK_DGRAM)
        self.data = None
        
        pass
    def run(self):
        self.running = True
        
        while self.running :
            self.data = self.s.recvfrom(1024)
            print(self.data)
            
        self.s.close()
        
    def senddat(self,dat):
        
        self.s.sendto(dat, self.address)
        
    def exit(self):
    
        self.running = False
            
            
class capThread(threading.Thread):
    def __init__(self, camera_path, img_height, img_width):
        super(capThread, self).__init__()
        self.camera_path = camera_path
        self.img_height = img_height
        self.img_width = img_width
        
        self.new_frame = False
        
        self.cap = cv2.VideoCapture(self.camera_path)
        
        ret , self.frame = self.cap.read()
        ret , self.hires_frame = self.cap.read()
        
    def has_new_frame(self):
        return self.new_frame
        
    def get_frame(self):
        self.new_frame = False
        return deepcopy(self.frame)
    def get_hiResFrame(self):
        self.new_frame = False
        return deepcopy(self.hires_frame)

    def run(self):
        global thread_exit
        
        while not thread_exit:
            ret, self.hires_frame = self.cap.read()
            if ret:
                frame = cv2.resize(self.hires_frame.copy(), (self.img_width, self.img_height))
                
                thread_lock.acquire()
                self.frame = frame
                self.new_frame = True
                thread_lock.release()

            else:
                #thread_exit = True
                print("cap error")
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.camera_path)
        self.cap.release()


class diff_firstFrame_tracker():
    firstframe = None
    frameCnt = 0
    
    def __init__(self):
        
        self.last_x = 0
        self.last_y = 0
        self.last_h = 0
        self.last_w = 0
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
        pass
    def init(self,frame, loc):

        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        gray=cv2.GaussianBlur(gray,(5,5),0)  
        self.firstframe = gray
    
        
    def update(self,frame):
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
        gray=cv2.GaussianBlur(gray,(5,5),0)  
        
        frameDelta = cv2.absdiff(self.firstframe,gray) 
        #frameDelta = cv2.subtract(gray, self.firstframe,gray)
        
        thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]  
        thresh = cv2.erode(thresh, self.kernel)
        thresh = cv2.dilate(thresh, self.kernel, iterations=3)  



        contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        
        
        M = 0
        if(len(contours) > 0):
            for i in contours:
                M = abs(cv2.contourArea(i))
                #x,y,w,h = cv2.boundingRect(i)
                #if(M < 40 or M > 600):
                #    thresh = cv2.rectangle(thresh, (x,y), (x+w,y+h), 0, thickness = cv2.FILLED)
                
        cv2.imshow("frameDelta", cv2.resize(frameDelta, (320,320)))         
        cv2.imshow("thresh", cv2.resize(thresh, (320,320)))
        
            
        if(M < 40 or M > 600):
            return True, (self.last_x, self.last_y, self.last_w, self.last_h)
            
        
        x,y,w,h = cv2.boundingRect(thresh)
        (self.last_x, self.last_y, self.last_w, self.last_h) = x,y,w,h 
        

        
        return True, (x,y,w,h)
        
        
        
class diffFrame_tracker():
    firstframe = None
    frameCnt = 0
    x=y=w=h = 0
    
    def __init__(self):
    
        pass
    def init(self, frame, _):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        gray=cv2.GaussianBlur(gray,(5,5),0)  
        self.firstframe = gray
        
        
    def update(self, frame):
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        gray=cv2.GaussianBlur(gray,(5,5),0)  
 

        self.frameCnt = self.frameCnt + 1

            
        frameDelta = cv2.absdiff(self.firstframe,gray)
        frameDelta = cv2.medianBlur(frameDelta, 3)
        
        thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]  
        thresh = cv2.dilate(thresh, None, iterations=2)  
        
        '''
        contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        M = 0
        if(len(contours) > 0):
            for i in contours:
                M = cv2.contourArea(i)
                
        if(M > 100 and M < 600):
            self.x,self.y,self.w,self.h = cv2.boundingRect(thresh)
        
        cv2.imshow("thresh", thresh)
        '''
        cx, xy, cw, ch = cv2.boundingRect(thresh)
        if(cw > 8 and ch > 20):
            (self.x,self.y,self.w,self.h) = cx, xy, cw, ch
        
        self.firstframe=gray
        
        return True, (self.x,self.y,self.w,self.h)




def main(net):
    global isFirstCap, firstCapMethod, ReCapMethod, frame
    ok = False
    
    pX = 0
    pY = 0
    reduceX = 5
    reduceY = 4

    failCnt = 0
    ret, frame = get_frame_()
    
    if(capFrameMethod == "f_frame_diff" or capFrameMethod == "frame_diff"):
        return True, (0,0,0,0), frame
    
    
    if(ReCapMethod == "nanodet"):
        print("nanodet")
        while not ok:
            #ret, frame = get_frame_()
            #frame = thread.get_hiResFrame()
            #frame = cv2.resize(frame, (320,320), interpolation = cv2.INTER_CUBIC)
            
            ret = False
            
            
            while ret == False:
                _, frame = get_frame_()
                #frame = thread.get_hiResFrame()
                #frame = cv2.resize(frame, (320,320), interpolation = cv2.INTER_CUBIC)
                #cv2.imshow('cam', frame)
                
                
                ret, xmin, ymin, xmax, ymax = net.det_one(frame)
                if(ret == False):
                    failCnt = failCnt + 1
                    if(failCnt > 4):
                        print("net not found.")
                        firstCapMethod = "frame_diff"
                        ReCapMethod = "frame_diff"
                        failCnt = 0
                        return False, (1,1,1,1),0
                    pass
    
            #xmin = int(xmin * (240/frame.shape[1]))
            #xmax = int(xmax * (240/frame.shape[1]))
            #ymin = int(ymin * (240/frame.shape[0]))
            #ymax = int(ymax * (240/frame.shape[0]))
            locations = (pX + xmin + reduceX , pY + ymin + reduceY, xmax - xmin - reduceX, ymax - ymin - reduceY)

            print(locations)
            ok = True
        #cv2.imshow("cap1",cv2.rectangle(frame.copy(), (xmin,ymin), (xmax,ymax), (0, 0, 255), 1, 1))
        #_, frame = get_frame_()
        #cv2.imshow("cam", cv2.resize(frame, (480,480), interpolation = cv2.INTER_CUBIC))
        
        return True, locations, frame    
    
    
    if(firstCapMethod == "frame_diff" or ReCapMethod == "frame_diff"):
        x,y,w,h = (0,0,0,0)
        x_,y_,w_,h_ = (0,0,0,0)
        print("Detecting...")
        failCnt = -1
        capSus = False
        while x < 10 or y < 5 :
            failCnt = failCnt + 1
            _, frame1 = get_frame_()
            
            gray1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)  
            gray1=cv2.GaussianBlur(gray1,(5,5),0) 
            
            _, frame2 = get_frame_()
            gray2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)  
            gray2=cv2.GaussianBlur(gray2,(5,5),0)       
            
            frameDelta = cv2.absdiff(gray1,gray2)  
            thresh = cv2.threshold(frameDelta, 40, 255, cv2.THRESH_BINARY)[1]  
            thresh = cv2.dilate(thresh, None, iterations=2)
            x,y,w,h=cv2.boundingRect(thresh)
            #cv2.imshow("cap1",cv2.rectangle(frame1.copy(), (x,y), (x+w,y+h), (0, 0, 255), 1, 1))
            #cv2.imshow("cap2",cv2.rectangle(frame2.copy(), (x,y), (x+w,y+h), (0, 0, 255), 1, 1))
            cv2.imshow("cam", cv2.resize(frame1, (480,480), interpolation = cv2.INTER_CUBIC))
            
            x_ = x + (w / 2)
            y_ = y + (h / 2)
            w_ = 10
            h_ = 15
            x_ = x_ - w_/2
            y_ = y_ - h_/2
            
            x_,y_,w_,h_ = int(x),int(y),int(w),int(h)
            capSus = True
            if(failCnt > 6):
                firstCapMethod = "nanodet"
                ReCapMethod = "nanodet"
                failCnt = 0
                capSus = False
                print("diff not found.")
                break
            
        #isFirstCap = False
        
        #cv2.imshow("cap2",cv2.rectangle(frame2.copy(), (x,y), (x+w,y+h), (0, 0, 255), 1, 1))
        if(capSus and (x_ > 0 or y_ > 0 or h_ > 0 or w_ > 0)):
            return True, (x_,y_,w_,h_ ), frame2
        else:
            
            return False, (0,0,0,0),0
        



def tracker_start():


    #tracker = cv2.TrackerCSRT_create()
    #tracker = cv2.legacy_TrackerMOSSE.create()
    #tracker = cv2.legacy.TrackerMOSSE_create()
    #tracker = cv2.legacy.TrackerBoosting_create()
    #tracker = cv2.legacy.TrackerMedianFlow_create()
    #tracker = cv2.TrackerKCF_create()
    #tracker = cv2.TrackerMIL_create()
    
    
    if(capFrameMethod == "f_frame_diff"):
        tracker = diff_firstFrame_tracker()
    if(capFrameMethod == "frame_diff"):
        tracker = diffFrame_tracker()
    if(capFrameMethod == "CSRT"):
        tracker = cv2.TrackerCSRT_create()
        
        
    
    return tracker


def move_filter(x, y):
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
    last_prediction = current_prediction # 把当前预测存储为上一次预测
    last_measurement = current_measurement # 把当前测量存储为上一次测量
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
    kalman.correct(current_measurement) # 用当前测量来校正卡尔曼滤波器
    current_prediction = kalman.predict() # 计算卡尔曼预测值，作为当前预测

    lmx, lmy = last_measurement[0], last_measurement[1] # 上一次测量坐标
    cmx, cmy = current_measurement[0], current_measurement[1] # 当前测量坐标
    lpx, lpy = last_prediction[0], last_prediction[1] # 上一次预测坐标
    cpx, cpy = current_prediction[0], current_prediction[1] # 当前预测坐标
    
    return lpx, lpy, cpx, cpy


def tracker_working(tracker):
    global fps_cnt
    global start_time
    global time_100ms
    global last_past_time
    global last_past_loc
    global line_length
    global g
    global CrossToLeftPeriod
    global CrossToRightPeriod
    global lastCrossToRightTime
    global lastCrossToLeftTime
    global CrossPeriod
    global detLine
    global detExceptionCnt
    global runingTime

    
    i = 0

    lastPeriod = 0
    min_Xc = detLine
    max_Xc = detLine
    max_X_len = 0
    
    line_length = 0
    
    
    crossTimes = 0
    
    while True:
        ret, frame = get_frame_()
        
        
        ok, new_location = tracker.update(frame)
        
        
        if ok:


            
            # Tracking success
            p1 = (int(new_location[0]), int(new_location[1]))
            p2 = (int(new_location[0] + new_location[2]),
                  int(new_location[1] + new_location[3]))
             
             
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
            
            
            x_mid = p1[0] + (p2[0] - p1[0])//2 
            y_mid = p1[1] + (p2[1] - p1[1])//2  
            
            _, _, x_mid_filt, y_mid_filt = move_filter(x_mid, y_mid)
            #x_mid_filt, y_mid_filt = (x_mid, y_mid)
            
            
            if x_mid_filt > 0 and y_mid_filt > 0 and x_mid_filt < 320 and y_mid_filt < 320:
                frame = cv2.circle(frame, (int(x_mid_filt), int(y_mid_filt)), 4, (255,0,0), 0 )
            
            if(x_mid_filt < detLine):
                now_loc = False
                if(x_mid_filt < min_Xc):
                    min_Xc = x_mid_filt
            else:
                now_loc = True
                if(x_mid_filt > max_Xc):
                    max_Xc = x_mid_filt
            
            if(now_loc != last_past_loc):
                
                if(now_loc == True and last_past_loc == False):     #From L to R
                    CrossToRightPeriod = (time.time() - lastCrossToRightTime)
                    lastCrossToRightTime = time.time()
                    #print("CTR Period:%f" % (CrossToRightPeriod))

                        
                if(now_loc == False and last_past_loc == True):     #From R to L
                    CrossToLeftPeriod = (time.time() - lastCrossToLeftTime)
                    lastCrossToLeftTime = time.time()
                    #print("CTL Period:%f" % (CrossToLeftPeriod))

                    
                last_past_loc = now_loc
                CrossPeriod = (CrossToLeftPeriod + CrossToRightPeriod) / 2
                print("Period:%f" % CrossPeriod)
                
                periodList[i] = CrossPeriod
                i = i + 1
                if(i > len(periodList) - 1):
                    i = 0
                
                crossTimes = crossTimes + 1
                
                if(max_Xc != detLine and min_Xc != detLine and crossTimes % 3 == 0):
                
                    max_X_len = max_Xc - min_Xc
                    print("max_Xc:%f, min_Xc:%f, X len max:%f" % ( max_Xc, min_Xc ,max_X_len))
                    
                    
                    
                    max_Xc = detLine
                    min_Xc = detLine
    


      
            fps_cnt = fps_cnt + 1
            recap = 0
            
            if time.time() - time_100ms >= 0.05:
                jsdat = json.dumps(
                        {
                            'id': 1,
                            'line_length':line_length, 
                            'loc': new_location,
                            'x_max_len': float(max_X_len)
                        }
                    ) 
                thread_client.senddat( jsdat.encode('UTF-8'))
                
                time_100ms = time.time()
            
            if time.time() - start_time >= 1:
            

                periodMean = np.median(periodList)
                
                if max_X_len > 30:
                    line_length =  g * (( periodMean / (2 * math.pi) ) ** 2) * 100  -  line_cut_off
                else:
                    line_length =  g * (( periodMean / (2 * math.pi) ) ** 2) * 100  -  line_cut_off2
                    
                if(CrossPeriod < 0.5 or CrossPeriod > 2.5 or lastPeriod == CrossPeriod
                        ):
                    
                    detExceptionCnt = detExceptionCnt + 1
                    if(detExceptionCnt == 3 and crossTimes > 3):
                        detLine = min_Xc + (max_Xc - min_Xc) / 2
                        
                    if(detExceptionCnt == 5 and capFrameMethod != "f_frame_diff" and capFrameMethod != "frame_diff"):
                        
                        print("Period exception, restart capture")
                        recap = 1
                        
                    if(detExceptionCnt >= 6):
                        detLine = 600/2
                        
                        detExceptionCnt = 0

                if(recap == 1):
                    detExceptionCnt = 6
                    break

                    
                    
                    
                lastPeriod = CrossPeriod  
                
                runingTime = runingTime + 1
                
                print("RT:%d, fps:%d, x_mid_filt:%d, y_mid:%d, T:%f, len: %f" % (runingTime, fps_cnt, x_mid_filt , y_mid , periodMean , line_length ))


                fps_cnt = 0
                start_time = time.time()
                
                
                
                
            
                
            

        else:


            break
            
        #frame = cv2.resize(frame, (320,320), interpolation = cv2.INTER_CUBIC)
        frame = cv2.line(frame, ( frame.shape[1]//2,0), (frame.shape[1]//2,frame.shape[0]), (0,0,255), 1) 
        
        frame = cv2.line(frame, ( int(detLine)  ,0), ( int(detLine)  ,frame.shape[0]), (255,0,0), 1) 
        #cv2.imshow("get ROI", frame)
        cv2.imshow("cam", cv2.resize(frame, (960,540), interpolation = cv2.INTER_CUBIC))
        
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        elif key == ord('q'):
            break
        elif key == ord('r'):
            runingTime = 0
            
    return True

def get_frame_():
    while(thread.has_new_frame() == False):
        time.sleep(0.01)
        
    thread_lock.acquire()
    frame = thread.get_frame()
    thread_lock.release()
    
    return True, frame

def judge_tracker(tracker, box, fr):
    #global frame
    
    #_ , frame = get_frame_()
    
    if(capFrameMethod == "f_frame_diff"):
        print("Cap First Frame...")
        input()
        _, fr = get_frame_()
        #print("wait 5s Preparing...")
        #time.sleep(5)
        
    
    
    tracker.init(fr, box)
    
    
    
    
if(capFrameMethod == "f_frame_diff" or capFrameMethod == "frame_diff"):
    thread = capThread("rtsp://admin:a1234567@192.168.4.161", 600, 600)
else:
    thread = capThread("rtsp://admin:a1234567@192.168.4.161/h264/ch1/sub/av_stream", 240, 240)



thread_client = clientThread("192.168.5.220",14400)

if __name__ == '__main__':


    #thread = capThread("rtsp://admin:a1234567@192.168.4.160", 240, 240)

    
    
    #thread = capThread("rtsp://admin:a1234567@192.168.5.120", 320, 320)
    thread.start()
    thread_client.start()
    
    while(thread.has_new_frame() == False):
        time.sleep(0.01)    
    
    net = my_nanodet(input_shape=320, prob_threshold=0.55, iou_threshold=0.8)
    
    
    ret = True
    
    while ret:
        r = False
        while r == False:
            r, location, _f = main(net)
        
        tracker = tracker_start()
        judge_tracker(tracker, location, _f)
        ret = tracker_working(tracker)
        
        
        
    cv2.destroyAllWindows()
    
    thread_client.exit()
    thread_exit = True
    
    
    