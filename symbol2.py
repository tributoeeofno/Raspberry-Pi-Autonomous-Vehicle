import cv2
import numpy as np


#Difference Variable
minSquareArea = 100000
width = 640
height = 480

ReferenceImage = ["symbol1.png", "symbol2.png","symbol3.png","symbol4.png"]
ReferenceTitles = ["Question", "Face","Traffic Light","No entry"]
templates = []

# Initialize variables to store the previous centroid positions
prev_centroid_x = 0
prev_centroid_y = 0
tip_y = 0
tip_x = 0
# Initialize a list to store the previous centroid positions for smoothing
centroid_history = []
# Number of previous positions to consider for smoothing
history_length = 500


def readRefImage(): #takes in template sample image and convert to blur 
    for count in range (len(ReferenceImage)):
        imagine_r = cv2.imread(ReferenceImage[count])
        image_r = cv2.cvtColor(imagine_r, cv2.COLOR_BGR2GRAY)
        blurred_r = cv2.GaussianBlur(image_r, (3, 3), 0)
        sample= cv2.resize(blurred_r, (int(width/2), int(height/2)), interpolation= cv2.INTER_AREA)
        templates.append(sample)

#captured vid process     
def order_points(pts): #takes in vertice points of quadilateral contour detected and return rect which is in a specific order
    rect = np.zeros((4,2), dtype = "float32")
    s = pts.sum(axis = 1) #calculate sum of x and y of vertice points 
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

#captured vid process
def four_point_transform(image, pts): #takes in frame and vertice points of contour detected and warp the image 
    rect = order_points(pts)

    maxWidth = int(width/2)
    maxHeight = int(height / 2)

    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped



def find_arrow_tip(contour, centroid, prev_centroid_x, prev_centroid_y):
    # Calculate Euclidean distances between centroid and contour points
    distances = [np.linalg.norm(point - centroid) for point in contour]
    

    # Find the index of the point with the maximum distance
    tip_index = np.argmax(distances)

    # If the distance from the previous centroid to the tip is shorter, use the previous centroid
    prev_tip_distance = np.linalg.norm(contour[tip_index][0] - (prev_centroid_x, prev_centroid_y))
    current_tip_distance = distances[tip_index]

    if prev_tip_distance < current_tip_distance:
        tip_coordinate = (prev_centroid_x, prev_centroid_y)
    else:
        # Return the coordinates of the tip
        tip_coordinate = tuple(contour[tip_index][0])

    # Limit the x value of the tip coordinate within 5 units of prev_centroid_x
    tip_x = tip_coordinate[0]
    
    # Update the tip coordinate with the limited x value
    tip_coordinate = (int(tip_x), int(tip_coordinate[1]))
    tip_x = tip_coordinate[0]
    tip_y = tip_coordinate[1]
    
    # Determine the direction based on the displacement #150 350
    if tip_x <320:
        if tip_y < 200:
         direction = "Up"
        elif tip_y > 280: 
         direction = "Down"
        else: 
         direction = "Right"
        
    elif tip_x >= 320:
      if tip_y < 200:
         direction = "Up"
      elif tip_y > 280: 
         direction = "Down"
      else:
         direction = "Left"


    return direction, tip_x, tip_y



def filter_contours(contours, min_area=10000, max_area=300000, max_sides=7, epsilon_factor=0.02):
    filtered_contours = []
    circleflg = 0  # flag to determine if shape is a circle
    partialcircleflg = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
        area = cv2.contourArea(contour)
        if min_area < area < max_area:#for polygons with edges
            if len(approx) <= max_sides:
                filtered_contours.append(contour)

            else:
                # Check if the contour is approximately a circle
                x, y, w, h = cv2.boundingRect(contour) #for regular size shapes
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.2:  #for circle
                    hull=cv2.convexHull(contour)
                    (x_val,y_val),radius=cv2.minEnclosingCircle(hull)
                    center=(int(x),int(y))
                    circle_area=np.pi*radius**2
                    circularity=area/circle_area
                    circularity_threshold=0.9
                    if circularity<circularity_threshold:
                        partialcircleflg = 1
                    else:
                        circleflg = 1
                  
                    filtered_contours.append(contour)

    return filtered_contours, circleflg, partialcircleflg

def determineSymbol(sorted_contours,sorted_contours_0,crop_img, thresh,circleflg, partialcircleflg):
    readRefImage()
    templateflg=False
    measure_distance=0
    face_detection=0
    for contour in sorted_contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_vertices=len(approx)
        area = cv2.contourArea(approx)
        if (num_vertices==4 and area>minSquareArea):
                cv2.drawContours(crop_img, [approx], 0, (0,255,255), 2)
                warped_eq = four_point_transform(thresh, approx.reshape(4,2)) #reshape the matrix until only 4 points are present
                max_res = -np.inf
                max_index = -1
                i = 0
                while i < len(ReferenceImage):#imagine taking a frame from camera and comparing with first template
                    res = cv2.matchTemplate(warped_eq, templates[i], cv2.TM_CCOEFF_NORMED) #res is an array containing different similarity scores at different parts of the frame, 
                    if np.any(res > max_res): #if there is a similar image in frame, res must be high
                        max_res = np.max(res) #get the maximum res
                        max_index = i
                        #when while loop iterates over the second template, res will never beat max_res if no other templates present in frame
                    i += 1
                if max_res is not None: 
                    if max_res > 0.10:
                        print("Template matching result", max_res)
                        print("Matching template", ReferenceTitles[max_index])
                        cv2.putText(crop_img, ReferenceTitles[max_index], (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                        templateflg=True
                        circleflg=-1
                        partialcircleflg=-1
                        if  ReferenceTitles[max_index] == "Face":
                            face_detection = 1
                        elif ReferenceTitles[max_index] == "Question":
                            measure_distance = 1
                        return(face_detection,measure_distance)
                    
    for contour in sorted_contours_0:
        if templateflg !=True:
            if circleflg == 1 and partialcircleflg == 0:
                num_vertices = 0  # Set number of vertices to 0 for circles
            elif circleflg == 0 and partialcircleflg == 1:
                num_vertices = 3
            else:
                if num_vertices==7:
                        circleflg=0
                        partialcircleflg=0
            cv2.drawContours(crop_img,[sorted_contours[0]], -1, (0, 255, 0), 3)  # Draw only the first contour
            writeNum(contour, crop_img, num_vertices, circleflg, partialcircleflg)
            return(face_detection,measure_distance)

def writeNum(contour, crop_img, num_vertices, circleflg, partialcircleflg):
    shapes = ['circle', 'partial_circle', 'triangle', 'rectangle/square', 'pentagon', 'hexagon','arrow']
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cy1 = int(M['m01'] / M['m00']) + 50 #this is just to print the vertices above the name
    
    if num_vertices == 0 and circleflg==1 and partialcircleflg==0:
        shape_index = 0  # Circle
    elif num_vertices > 0 and num_vertices <= 7:
        shape_index = min(num_vertices - 1, len(shapes) - 1)
        if circleflg == 0 and partialcircleflg == 1 and num_vertices==3: #consider partial circle which has three vertices
            shape_index = 1
        if num_vertices==7:
            direction, tip_x, tip_y = find_arrow_tip(contour, (cx, cy), prev_centroid_x, prev_centroid_y)
            cv2.circle(crop_img, (tip_x, tip_y),1,(0,0,255),5)
            if direction is not None:
               shapes[6]+=" "+direction
        
    else:
        shape_index = -1  # Unknown shape

    if shape_index != -1:
         print(shapes[shape_index])
    else:
         print('unknown shape')