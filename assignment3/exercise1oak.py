import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt
import depthai as dai
import time

"""
Just a small script trying a bit around using a OAK-D Lite
"""

def showimgs(img1, img2, img3, size=(960,540), text='imgs'):
    #showimg = np.concatenate((cv.resize(img1, size), cv.resize(img2, size), cv.resize(img3, size)), axis=1)
    showimg = np.concatenate((img1, img2, img3), axis=1)
    cv.imshow(text, showimg)

def calc_disparity(img1, img2):
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=21)
    #stereo.setTextureThreshold(1000)
    #stereo.setSpeckleRange(1000)
    stereo.setUniquenessRatio(100000)
    return stereo.compute(img1, img2)

# parser = argparse.ArgumentParser(description='Demstration image disparity')
# parser.add_argument('video', type=str, help='path to video file')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.video)
# ret, frame1 = cap.read()
# frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# for i in range(10):
#     ret, frame2 = cap.read()
#     if not ret:
#         print("too few images!")
# frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

# disp1 = calc_disparity(frame1, frame2)
# #showimgs(frame1, frame2, disp1, text='nearby')
# plt.imshow(disp1,'gray')
# plt.show()
# for i in range(50):
#     ret, frame3 = cap.read()
#     if not ret:
#         print("too few images!")
# frame3 = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
# disp2 = calc_disparity(frame1, frame3)
# #showimgs(frame1, frame3, disp2, text='far')
# plt.imshow(disp2,'gray')
# plt.show()

def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()

    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else :
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

if __name__ == '__main__':
    pipeline = dai.Pipeline()
 
    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)
 
    # Set output Xlink for left camera
    xoutLeft = pipeline.createXLinkOut()
    xoutLeft.setStreamName("left")
 
    # Set output Xlink for right camera
    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName("right")
  
    # Attach cameras to output Xlink
    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)

    with dai.Device(pipeline) as device:
        # Get output queues. 
        leftQueue = device.getOutputQueue(name="left", maxSize=1)
        rightQueue = device.getOutputQueue(name="right", maxSize=1)
    
        # Set display window name
        cv.namedWindow("Stereo Pair")
        # Variable used to toggle between side by side view and one frame view. 
        sideBySide = True

        while True:
            # Get left frame
            leftFrame = getFrame(leftQueue)
            # Get right frame 
            rightFrame = getFrame(rightQueue)
            disp_img = calc_disparity(leftFrame, rightFrame)
            rightFrame = (rightFrame/np.max(rightFrame))
            leftFrame = (leftFrame/np.max(leftFrame))
            disp_img = (disp_img/np.max(disp_img))
            if sideBySide:
                # Show side by side view
                imOut = np.hstack((leftFrame, rightFrame, disp_img))
            else : 
                # Show overlapping frames
                imOut = np.uint8(leftFrame/2 + rightFrame/2)
            # plt.imshow(imOut,'gray')
            # plt.show()
            # Display output image
            cv.imshow("Stereo Pair", imOut)
            
            # Check for keyboard input
            key = cv.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            # elif key == ord('t'):
            #     # Toggle display when t is pressed
            #     sideBySide = not sideBySide