 
import numpy as np
import cv2 as cv
import argparse
import sys
import time

def read_grayscale_img(cap, avg_img):
    ret, img = cap.read()
    if not ret:
        print('Stream ended!')
        cap.release()
        sys.exit(0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (15,15), 1.0)
    return cv.normalize(img - cv.GaussianBlur(avg_img, (15,15), 1.0), None, 0, 255, cv.NORM_MINMAX)

def apply_separable_filter(img, filter):
    img = cv.filter2D(img, -1, filter)
    return cv.filter2D(img, -1, filter.T)

def make_avg_img(path, limit=25000):
    cap2 = cv.VideoCapture(path)
    ret, img = cap2.read()
    if not ret:
            cap2.release()
            return img
    cnt = 0
    avg = np.zeros(cv.cvtColor(img, cv.COLOR_BGR2GRAY).shape, dtype=np.float64)
    while cnt<limit:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        avg += img
        cnt+=1
        ret, img = cap2.read()
        if not ret:
            print('Avg done')
            cap2.release()
            return np.uint8(avg/cnt)
    print('Avg done')
    cap2.release()
    return np.uint8(avg/cnt)

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='path to video file')
args = parser.parse_args()
# fourcc = cv.VideoWriter_fourcc(*'mp4v') 
# videosaver = cv.VideoWriter(args.video.replace(".mp4", "")+'spatiotemporal2.mp4',cv.CAP_FFMPEG, fourcc,24,(1280,720), isColor=False)


avg_img = make_avg_img(args.video)
cv.imshow("", avg_img)
cap = cv.VideoCapture(args.video)

filter_x = np.array([0.0094, 0.1148, 0.3964, -0.0601, -0.9213, -0.0601, 0.3964, 0.1148, 0.0094])
filter_t = np.array([0.0008, 0.0176, 0.1660, 0.6383, 1.0, 0.6383, 0.1660, 0.0176, 0.0008])

imgs = []

#load 9 (space convolved) images into queue
for i in range(9):
    img = read_grayscale_img(cap, avg_img)
    imgs.append(apply_separable_filter(img, filter_x))  
    
while(1):
    dimensions = np.array(imgs).shape
    t_convolved = np.zeros((dimensions[1], dimensions[2]))

    #calculate the convolution of the images
    for i, time_filter in enumerate(filter_t):
        t_convolved += imgs[i] * time_filter
    
    #normalize and convert to 8bit int
    out = (t_convolved / np.max(t_convolved)) * 255
    out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX)
    out = np.uint8(out)
    

    #remove oldest image and add a new one
    imgs.pop(0)
    img = read_grayscale_img(cap, avg_img)
    imgs.append(apply_separable_filter(img, filter_x))

    # videosaver.write(cv.resize(out, (1280,720)))
    cv.imshow('fr', out)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# videosaver.release()