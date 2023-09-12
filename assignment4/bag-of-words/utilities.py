import cv2 as cv
import math
import numpy as np
import tqdm 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_test_split_data(X, y, factor=0.8, random=True):
    assert len(X) == len(y)
    size = len(X)
    #size_train = math.floor(factor * size)
    site_test = math.ceil((1-factor) * size)
    #shuffle
    Xn = [None] * size
    yn = [None] * size
    if random:
        permutation = np.random.permutation(size)
        for old_index, new_index in enumerate(permutation):
            Xn[new_index] = X[old_index]
            yn[new_index] = y[old_index]
    else:
        Xn = X
        yn = y

    # split
    X_test_split = Xn[:site_test]
    X_train_split = Xn[site_test:]
    y_test_split = yn[:site_test]
    y_train_split = yn[site_test:]
    return X_train_split, X_test_split, y_train_split, y_test_split

def flatten(arr):
    if(isinstance(arr[0], np.ndarray) or isinstance(arr[0], list)):
        arr = np.array([item for sublist in arr for item in sublist])
    return arr

def select_random_elements(arr, max_elements=5, random=True):
    unique_list = list(set(arr))
    # if len(unique_list) < max_elements:
    #     return arr
    if random:
        np.random.shuffle(unique_list)
    new_list = unique_list[:max_elements]
    return new_list

def label_encoding(y, unique_y):
    y = np.array(y)
    unique_y = np.array(unique_y)
    ys = []
    # convert y labels into an integer
    for s in y:
        ys.append(np.min(np.nonzero(unique_y == s)))
    return ys

def print_prediction(y_prediction, y):
    correct_array = np.equal(y_prediction, y)
    tot = len(y)
    percentage = (np.sum(correct_array)/tot) * 100
    print(f'num of correct  {np.sum(correct_array)} total: {tot} percentage: {percentage}')
    return percentage

def show_confusion_mat(y_true, y_pred):
    ## confusion mat
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred
    )
    disp.ax_.set_title("Confusion Matrix")
    print(disp.confusion_matrix)
    plt.show()

