import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Epipolar Geometry')
parser.add_argument('--img1', type=str, default='./input/exercise1out1.png', help='path to first image file')
parser.add_argument('--img2', type=str, default='./input/exercise1out2.png', help='path to second image file')
parser.add_argument('--useRansac', type=int, default=0, help='use RANSAC (1) or LMEDS (0)')
parser.add_argument('--save', type=int, default=0, help='store image (1) or don\'t (0)')
parser.add_argument('--warp', type=int, default=0, help='show/save the warped image')
args = parser.parse_args()

# img1 = cv.imread('./PhotoTurismChallenge/pragueparks/lizard/frame000071.jpg', cv.IMREAD_GRAYSCALE)  #queryimage # left image
# img2 = cv.imread('./PhotoTurismChallenge/pragueparks/lizard/frame000073.jpg', cv.IMREAD_GRAYSCALE) #trainimage # right image
img1 = cv.imread(args.img1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(args.img2, cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
if args.useRansac == 1:
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC)
else:
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
if args.save == 1:
    ransac_lmeds_str ='_ransac' if args.useRansac == 1 else '_lmeds'
    cv.imwrite(args.img1.replace(".png", "")+'ex2_sift'+ ransac_lmeds_str +'.png', np.hstack((img5, img3)))

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

if args.warp == 1:
    imsize = np.array(img1).shape
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, imsize)
    corrected_img1 = cv.warpPerspective(img1, H1, imsize)
    corrected_img2 = cv.warpPerspective(img2, H2, imsize)

    if args.save == 1:
        ransac_lmeds_str ='_ransac' if args.useRansac == 1 else '_lmeds'
        cv.imwrite(args.img1.replace(".png", "")+'ex2_sift_warped'+ ransac_lmeds_str +'.png', np.hstack((corrected_img1, corrected_img2)))
    plt.subplot(121),plt.imshow(corrected_img1, 'gray')
    plt.subplot(122),plt.imshow(corrected_img2, 'gray')
    plt.show()
