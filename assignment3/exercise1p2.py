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

def findKpDes(img):
    sift = cv.SIFT_create()
    return sift.detectAndCompute(img, None) #returns kp, des

#https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
#https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
#using flann as it seems to report a greater num of matches 
def matchKp(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    print(np.array(matches).size)
    #filter for good matches
    matches_im1 = []
    matches_im2 = []
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matches_im2.append(kp2[m.trainIdx].pt)
            matches_im1.append(kp1[m.queryIdx].pt)
    return matches_im1, matches_im2

parser = argparse.ArgumentParser(description='Demstration image disparity')
parser.add_argument('--video', type=str, help='path to video file')
parser.add_argument('--save', type=int, default=0, help='store image (1) or don\'t (0)')
args = parser.parse_args()
cap = cv.VideoCapture(args.video)
ret, frame1 = cap.read()
frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# if args.save == 1:
#     cv.imwrite(args.video.replace(".mp4", "")+'outimg1.png', frame1)
for i in range(1):
    ret, frame2 = cap.read()
    if not ret:
        print("too few images!")
frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
# if args.save == 1:
#     cv.imwrite(args.video.replace(".mp4", "")+'outimg2.png', frame2)

disp1 = calc_disparity(frame1, frame2)
# if args.save == 1:
#     cv.imwrite(args.video.replace(".mp4", "")+'disparity_near.png', disp1)

plt.imshow(disp1,'gray')
plt.show()
for i in range(11):
    ret, frame3 = cap.read()
    if not ret:
        print("too few images!")
frame3 = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
# if args.save == 1:
#     cv.imwrite(args.video.replace(".mp4", "")+'outimg3.png', frame3)

disp2 = calc_disparity(frame1, frame3)
# if args.save == 1:
#     cv.imwrite(args.video.replace(".mp4", "")+'disparity_far.png', disp2)

plt.imshow(disp2,'gray')
plt.show()

disp = np.hstack((disp1, disp2))
# if args.save == 1:
#     cv.imwrite(args.video.replace(".mp4", "")+'disparity.png', disp)
plt.imshow(disp, "gray")
plt.show()


############## PART 2

img1 = frame1
img2 = frame3

kp1, des1 = findKpDes(img1)
kp2, des2 = findKpDes(img2)

match1, match2 = matchKp(des1, des2)
#https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(match1)
pts2 = np.int32(match2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

###
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
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'epilines.png', np.hstack((img5,img3)))
###

retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (1920, 1080))

#warp images accordingly

corrected_img1 = cv.warpPerspective(img1, H1, (1920, 1080))
corrected_img2 = cv.warpPerspective(img2, H2, (1920, 1080))
if args.save == 1:
    cv.imwrite(args.video.replace(".mp4", "")+'warped.png', np.hstack((corrected_img1,corrected_img2)))
#####

plt.subplot(121),plt.imshow(corrected_img1, "gray")
plt.subplot(122),plt.imshow(corrected_img2, "gray")
plt.show()
#########