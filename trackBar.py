import cv2
import numpy as np


#just for passing sake
def passfunction():
    pass

def initialiseTrackbars(parameter,init_val1,init_val2):
    #initialise trackbars
    cv2.namedWindow("Trackbars")
    i=0
    for i in range (len(parameter)):
        cv2.createTrackbar(parameter[i],"Trackbars",init_val1,init_val2,passfunction)


def getTrackbarPos(parameter):
    
    i=0
    retVals=[]
    for param in parameter:
        retVals.append(cv2.getTrackbarPos(param, "Trackbars"))
    
    return (retVals)
