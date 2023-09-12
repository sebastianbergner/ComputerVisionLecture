import numpy as np
import cv2 as cv
import argparse
import sys
import time

def read_grayscale_img(cap):
    ret, img = cap.read()
    if not ret:
        print('Stream ended!')
        cap.release()
        sys.exit(0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def apply_separable_filter(img, filter):
    img = cv.filter2D(img, -1, filter)
    return cv.filter2D(img, -1, filter.T)

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='path to video file')
args = parser.parse_args()
cap = cv.VideoCapture(args.video)
# fourcc = cv.VideoWriter_fourcc(*'mp4v') 
# videosaver = cv.VideoWriter(args.video.replace(".mp4", "")+'spatiotemporal1.mp4',cv.CAP_FFMPEG, fourcc,24,(1280,720), isColor=False)

filter_x = np.array([0.0094, 0.1148, 0.3964, -0.0601, -0.9213, -0.0601, 0.3964, 0.1148, 0.0094])
filter_t = np.array([0.0008, 0.0176, 0.1660, 0.6383, 1.0, 0.6383, 0.1660, 0.0176, 0.0008])

imgs = []

#load 9 (space convolved) images into queue
for i in range(9):
    img = read_grayscale_img(cap)
    imgs.append(apply_separable_filter(img, filter_x))  
    
while(1):
    dimensions = np.array(imgs).shape
    t_convolved = np.zeros((dimensions[1], dimensions[2]))

    #calculate the convolution of the images
    for i, time_filter in enumerate(filter_t):
        t_convolved += imgs[i] * time_filter
    
    #normalize and convert to 8bit int
    out = (t_convolved / np.max(t_convolved)) * 255
    out = np.uint8(out)

    #remove oldest image and add a new one
    imgs.pop(0)
    img = read_grayscale_img(cap)
    imgs.append(apply_separable_filter(img, filter_x))

    cv.imshow('fr', out)
    # videosaver.write(cv.resize(out, (1280,720)))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# videosaver.release()