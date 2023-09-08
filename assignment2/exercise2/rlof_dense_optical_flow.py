import numpy as np
import cv2 as cv
import argparse

def resize(img):
    return cv.resize(img, (720, 568))

parser = argparse.ArgumentParser(description='This sample demonstrates RLOF Dense Optical Flow calculation.')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()
frame1 = resize(frame1)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
# fourcc = cv.VideoWriter_fourcc(*'mp4v') 
# videosaver = cv.VideoWriter(args.image.replace(".mp4", "")+'_rlof.mp4',fourcc,1,(3840,1080))

while(1):
    ret, frame2 = cap.read()
    frame2 = resize(frame2)
    flow = cv.optflow.calcOpticalFlowDenseRLOF(frame1, frame2, None)

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    showimg = np.concatenate((cv.resize(bgr, (1920, 1080)), cv.resize(frame2, (1920, 1080))), axis=1)
    cv.imshow('flow', showimg)
    # videosaver.write(showimg)
    if cv.waitKey(1) & 0xFF == ord('q'):
        # videosaver.release()
        break
    frame1 = frame2
cv.destroyAllWindows()


