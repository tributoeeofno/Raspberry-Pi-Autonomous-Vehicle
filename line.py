import RPi.GPIO as GPIO
import cv2
import numpy as np
import math

   
y_max=210

IN1=37
IN2=35
IN3=33
IN4=31
enA=29
enB=32
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(enA,GPIO.OUT)
GPIO.setup(enB,GPIO.OUT)
GPIO.setup(IN1,GPIO.OUT)
GPIO.setup(IN2,GPIO.OUT)
GPIO.setup(IN3,GPIO.OUT)
GPIO.setup(IN4,GPIO.OUT)

PWM1=GPIO.PWM(enA,100)
PWM2=GPIO.PWM(enB,100)

def measure_distance(initial,final):
    speed=5.17
    distance=speed*(final-initial)
    return (distance)

def colour_Picker(c1,c2,c3):
    colour_Array=[[[39,100,81],[83,255,255]],[[170,0,0],[182,255,255]],[[90,0,0],[140,255,255]],[[25,192,88],[35,255,255]],[[0,0,0],[180,255,44]]]
    colour=['G','R','B','Y','K'] #K is black
    param_1=0
    param_2=0
    param_3=0
    for c in colour:
        if c==c1:
            param_1=colour_Array[colour.index(c)]
        if c==c2:
            param_2=colour_Array[colour.index(c)]
        if c==c3:
            param_3=colour_Array[colour.index(c)]
    
    return (param_1,param_2,param_3)
            
            
def movement(direction,status,durationWalk,durationStop,*args):# 1: direction #2 : change speed? #3: moving time #4 stopping time #5PWM value
        i=1
        #close all motors
        GPIO.output(IN1,False)
        GPIO.output(IN2,False)
        GPIO.output(IN3,False)
        GPIO.output(IN4,False)
        
        if direction=='F':
            logic=[0,1,1,0]
        elif direction=='L':
            logic=[1,0,1,0]
        elif direction=='R':
            logic=[0,1,0,1]
        elif direction=='B':
            logic=[1,0,0,1]
        elif direction=='S':
            logic=[0,0,0,0]
        
            
        PWM1.start(args[0])
        PWM2.start(args[0])
        GPIO.output(IN1,logic[0])
        GPIO.output(IN2,logic[1])
        GPIO.output(IN3,logic[2])
        GPIO.output(IN4,logic[3])

        if status==0:
            GPIO.output(IN1,0)
            GPIO.output(IN2,0)
            GPIO.output(IN3,0)
            GPIO.output(IN4,0)
    
def line_Detect(hsv,frame,c1,c2,c3):
    param_1,param_2,param_3=colour_Picker(c1,c2,c3)
   
    # create a mask for stated colours
    m1 = cv2.inRange(hsv,np.array(param_1[0]),np.array(param_1[1]))
    m2 = cv2.inRange(hsv,np.array(param_2[0]),np.array(param_2[1]))
    m3 = cv2.inRange(hsv,np.array(param_3[0]),np.array(param_3[1]))
    #dilate the mask
    kernel=np.ones((50,50),"uint8")
    color_m1=cv2.dilate(m1,kernel)
    color_m2=cv2.dilate(m2,kernel)
    color_m3=cv2.dilate(m3,kernel)
    #conduct masking on frame
    res_m1 = cv2.bitwise_and(frame, frame, mask=color_m1)
    res_m2 = cv2.bitwise_and(frame, frame, mask=color_m2)
    res_m3 = cv2.bitwise_and(frame, frame, mask=color_m3)
    # find contours
    contours_m1, _ = cv2.findContours(m1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_m2, _ = cv2.findContours(m2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_m3, _ = cv2.findContours(m3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours_m1) > 0 or len(contours_m2) > 0 or len(contours_m3) > 0:
        if len(contours_m1) > 0 or len(contours_m2)>0:
            if len(contours_m1)>0:
                contours = contours_m1
                color = c1
            else:
                contours = contours_m2
                color = c2
                
        elif len(contours_m3) > 0:
            contours = contours_m3
            color = c3
            
        line_detected = True

    else:
        print("No line detected. Stopping.")
        line_detected = False
        movement('B',2,0,0,24)
    
    # Process line following logic if a line is detected
    if line_detected == True:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, c, -1, (0, 0, 255), 3)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Implement your movement logic here based on centroid position (cx, cy)
            if cy < y_max:
                if cx >= 220 :
                    print("Turn Right",color)
                    movement('R',2,0,0,76)
                if cx < 220 and cx >110 :
                    print("On Track!",color)
                    movement('F',2,0,0,28)
                if cx <=110 :
                    print("Turn Left",color)
                    movement('L',2,0,0,56)
            if cy>=y_max:
                print("Special turn")
                if cx>=140:
                    print('special turn right',color)
                    movement('R',2,0,0,81)
                    
                if cx < 140 and cx >125 :
                    print("Special On Track!",color)
                    movement('B',2,0,0,50)

                if cx <125 :
                    print("special turn left",color)
                    movement('L',2,0,0,93)
                    
            print(f"CX : {cx} CY : {cy}")
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)