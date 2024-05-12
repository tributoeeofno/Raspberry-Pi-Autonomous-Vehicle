import time
import RPi.GPIO as GPIO
import cv2
import numpy as np
import math
import line as l
import symbol2 as s
c1='Y'
c2='B'
c3='K'

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)


def main ():
    
    face_detection=0
    measure_distance=0
    
    while True:
          ret, frame = cap.read()
          symbol_frame=cv2.resize(frame, (640, 480))
          #line detection
          hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          #symbol detect
          grey = cv2.cvtColor(symbol_frame, cv2.COLOR_BGR2GRAY)
          blur = cv2.GaussianBlur(grey, (3, 3), 0)
          ret,thresh = cv2.threshold(blur,98, 255, cv2.THRESH_BINARY)
          edges = cv2.Canny(thresh, 100, 200)
          contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
          filtered_contours, circleflg, partialcircleflg = s.filter_contours(contours)
          sorted_contours = sorted(filtered_contours, key=cv2.contourArea)

    
          if sorted_contours:  # Check if sorted_contours is not empty
            # Stop the car
            print("Contours detected. Stopping the car.")
            l.movement('S', 0, 0, 0, 0)
            
            # Loop for a specific number of iterations (e.g., 100 iterations)
            
            for _ in range(15):
                print("symbol detection mode.")
                # Display contours and determine symbols
                cv2.drawContours(symbol_frame, sorted_contours, 0, (0, 255, 0), 3)
                face_detection, measure_distance = s.determineSymbol(sorted_contours, [sorted_contours[0]], symbol_frame, thresh, circleflg, partialcircleflg)
                if face_detection==1:
                    break
                elif measure_distance==1:
                    initial=time.time()
            if face_detection==1:
                break
            

          else:
              # No contours detected, continue line detection
              print('line following mode.')
              l.line_Detect(hsv, frame,c1,c2,c3)
    if face_detection ==1:
        cap.release()
        final=time.time()
        distance=l.measure_distance(initial,final)
        print("distance estimated: ",distance,"cm")
        import face_recognition_ as facial
        facial.facial_recognition()
        
        

       

    
main()