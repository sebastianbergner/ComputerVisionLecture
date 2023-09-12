import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt

def calc_disparity(img1, img2):
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=25)
    disparity_map = stereo.compute(img1, img2)
    return disparity_map
    #normalized_disparity_map = cv.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #return normalized_disparity_map

parser = argparse.ArgumentParser(description='Demstration image disparity')
parser.add_argument('--video', type=str, help='path to video file')
parser.add_argument('--save', type=int, default=0, help='store image (1) or don\'t (0)')
args = parser.parse_args()
cap = cv.VideoCapture(args.video)
ret, frame1 = cap.read()
frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'outimg1.png', frame1)
for i in range(1):
    ret, frame2 = cap.read()
    if not ret:
        print("too few images!")
frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'outimg2.png', frame2)

disp1 = calc_disparity(frame1, frame2)
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'disparity_near.png', disp1)

plt.imshow(disp1,'gray')
plt.show()
for i in range(5):
    ret, frame3 = cap.read()
    if not ret:
        print("too few images!")
frame3 = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'outimg3.png', frame3)

disp2 = calc_disparity(frame1, frame3)
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'disparity_far.png', disp2)

plt.imshow(disp2,'gray')
plt.show()

disp = np.hstack((disp1, disp2))
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'disparity.png', disp)
plt.imshow(disp, "gray")
plt.show()