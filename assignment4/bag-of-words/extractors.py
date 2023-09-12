import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# shi_tomasi
def extract_descriptors_shi_tomasi(cvimgs, maxCorners=160, qualityLevel=0.2, minDistance=3, blockSize=15):
    extractor = cv.SIFT_create()
    extracted_descriptors = []
    for image in cvimgs:
        #detect kp
        kp_pos = cv.goodFeaturesToTrack(image, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)
        kp_pos = np.int0(kp_pos)
        kp = []
        for kpt in kp_pos:
            x,y = kpt.ravel()
            kp.append(cv.KeyPoint(float(x), float(y), 3.0))
        
        _, descriptors = extractor.compute(image, kp)
        if descriptors is None:
            descriptors = []
        extracted_descriptors.append(descriptors)
    return extracted_descriptors, cvimgs

# hog
def extract_fvs_hog(cvimgs, winSize = (32, 32), blockSize = (4,4), blockStride = (2,2), cellSize = (4,4), nbins = 6):
    cvimgs = [cv.resize(img, (128, 128)) for img in cvimgs]
    print("extract_keypoints_hog")
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_fv = [hog.compute(image) for image in cvimgs]
    return hog_fv, cvimgs


# features
def calc_cluster_centers(descriptors, n_classes=12):
    n_cluster = n_classes * 10
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers=cv.kmeans(descriptors, n_cluster, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    return centers


def calc_bow(descriptors_per_image, centers):
    n_cluster, n_points = centers.shape
    features = np.array([0.0] * n_cluster)
    for i in range(len(descriptors_per_image)):
        distances = np.sqrt(np.sum((centers-descriptors_per_image[i])**2, axis=1))
        min_pos = np.argmin(distances)
        features[min_pos] += 1 / len(descriptors_per_image)
    return features
