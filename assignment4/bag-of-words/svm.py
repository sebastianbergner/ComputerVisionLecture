import cv2 as cv
import numpy as np

# svm
def create_svm(k = cv.ml.SVM_RBF,c = 0.000001, g = 1):
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(k)
    svm.setC(c)
    svm.setGamma(g)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 5000000, 1e-7))
    return svm

def create_nu_svm(nu = 0.001, k=cv.ml.SVM_RBF, g=1):
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_NU_SVC)
    svm.setNu(nu)
    svm.setKernel(k)
    svm.setGamma(g)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 5000000, 1e-7))
    return svm

def train_svm(svm, X, y):
    #fix encoding
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("train SVM")
    svm.train(X, cv.ml.ROW_SAMPLE, y)
    return svm

def predict_multiple(svm, X):
    X = np.array(X, dtype=np.float32)
    print(f'Is svm trained {svm.isTrained()}')
    res = svm.predict(X)[1]
    return res.flatten()
