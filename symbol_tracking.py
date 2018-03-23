#found the largest contour and found its topmost, leftmost and righmost coordinates
# arrange corner points in clockwise fashion. Tutorial on pyimagesearch
import cv2
import numpy as np
import imutils

# image = cv2.imread('/Users/admin/Downloads/shape-detection/triangle_u.jpg')
image = cv2.VideoCapture(0)
# lower_range = (42, 90, 37)
# upper_range = (89, 255, 255)
lower_range = (32, 48, 48)
upper_range = (89, 255, 255)
while True:
    (grabbed,frame) = image.read()
    resized = imutils.resize(frame, width=900)
    ratio = frame.shape[0] / float(frame.shape[0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(gray, lower_range, upper_range)
    # mask = cv2.erode(mask, None, iterations=1)
    # mask = cv2.dilate(mask, None, iterations=1)
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # im = cv2.filter2D(mask, -1, kernel)
    # cv2.imshow("erode",mask)

    # test = cv2.Laplacian(mask, cv2.CV_32F)
    # cv2.imshow('laplace', test)
    # blur = cv2.GaussianBlur(test,(3,3),0)

    # # smooth = cv2.addWeighted(blur,1.5,mask,-0.5,0)

    # cv2.imshow('smooth',blur)
    # # gray = np.float32(test)
    # gray = np.float32(blur)

# threshold image
    ret, thresh = cv2.threshold(mask,127,255,0)
    

# filtering image
    median = cv2.medianBlur(thresh,3)
    bilateral = cv2.bilateralFilter(thresh,9,75,75)
    blur = cv2.blur(bilateral,(5,5))
    # mask = cv2.erode(blur, None, iterations=6)
    # mask = cv2.dilate(mask, None, iterations=6)
    blur = mask

# show region of interest in actual color
    result = cv2.bitwise_and(frame, frame, mask = median)
    cv2.imshow("filtered_result",result)
    cv2.imshow("median",median)
    cv2.imshow("bilateral",bilateral)
##########################
    # cnts = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    # center = None
    # if len(cnts)>0:
    #     c = max(cnts, key=cv2.contourArea)
    #     # ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     # M = cv2.moments(c)
    #     # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #     # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
    #     # cv2.circle(frame, center, 5, (0, 0, 255), -1)
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
############################
# find contours
 # im2, contours, hierarchy = cv2.findContours(bilateral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if(len(contours)>0):
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)
            cntx1 = contours[0][0]
            # cntx = contours[0][1]
            pt1 = (cntx1[0][0],cntx1[0][1])
            # pt2 = (cntx[0][0],cntx[0][1])
            cv2.circle(frame,pt1,5,(0,255,0),-1)
            # cv2.circle(frame,pt2, 5, (0,255,0),-1)
    else:
        continue
##################################
    c_max = max(contours, key=cv2.contourArea)
    # determine the most extreme points along the contour
    extLeft = tuple(c_max[c_max[:, :, 0].argmin()][0])
    extRight = tuple(c_max[c_max[:, :, 0].argmax()][0])
    extTop = tuple(c_max[c_max[:, :, 1].argmin()][0])
    extBot = tuple(c_max[c_max[:, :, 1].argmax()][0])
    cv2.circle(frame, extLeft, 3,(100,100,100), -1)
    cv2.circle(frame, extRight, 3,(10,100,100), -1)
    cv2.circle(frame, extTop, 3,(100,0,100), -1)
    # cv2.circle(frame, extBot, 3,(100,100,0), -1)
    print(extLeft, extTop, extRight)
###################################
#convert to float 32 image for detecting corners
    gray = np.float32(blur)
    # cv2.imshow('float',gray)

# detect corners
    corners = cv2.goodFeaturesToTrack(gray, 3, 0.01, 10)
    if corners is None:
        continue
    else:
        corners = np.int0(corners)
        count = 0
        for corner in corners:
            x,y = corner.ravel()
            # print(count,x,y)
            count+=1
            cv2.circle(frame, (x,y), 3, (230,120,110), -1)

    cv2.imshow('Corner', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
image.release()
cv2.destroyAllWindows()