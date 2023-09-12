import cv2 as cv
import dataloader
import utilities
import datasaver
import extractors
import svm
import numpy as np
import sys
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate(zip_file, save_file_additional, save_file_svm):
    preview_enabled = True
    
    save_data = datasaver.load_data(save_file_additional)

    if save_data is None:
        print("save_file_additional is not defined")
        return

    print("get_filelist_from_zip")
    filelist = dataloader.get_filelist_from_zip(zip_file)

    print("extract_all_categories_from_path")
    categorielist, filelist = dataloader.extract_all_categories_from_path(filelist, 1)

    categorielist_reduced = save_data["categorielist"]
    print("categorielist:", categorielist_reduced)

    print("limit_list_by_categories")
    categorielist, filelist = dataloader.limit_by_categories(categorielist,filelist, categorielist_reduced)

    print("label_encoding")
    label_encoding = utilities.label_encoding(categorielist,categorielist_reduced)

    print("import_images_from_zip")
    cv_images = dataloader.import_images_from_zip(zip_file, filelist, cv.IMREAD_GRAYSCALE)

    extract = save_data["extract"]
    if extract == "sift":
        print("extract_descriptors_sift")
        images_descriptors, cv_images = extractors.extract_descriptors_sift(cv_images)
    elif extract == "surf":
        print("extract_descriptors_surf")
        images_descriptors, cv_images = extractors.extract_descriptors_surf(cv_images)
    elif extract == "hog":
        print("extract_descriptors_hog")
        images_descriptors, cv_images = extractors.extract_fvs_hog(cv_images)
        images_descriptors = np.squeeze(images_descriptors)
    else:
        print("extract", extract, "not supported")
        return

    descriptors_center = save_data["descriptors_center"]

    print("calc_bow")
    features = [extractors.calc_bow(descriptors, descriptors_center) for descriptors in images_descriptors]

    svm_net = cv.ml.SVM_load(save_file_svm)
    y_predicted = svm.predict_multiple(svm_net, features)
    utilities.print_prediction(y_predicted, label_encoding)

    if preview_enabled: 
        font = cv.FONT_HERSHEY_SIMPLEX
        previewimgs = []
        for i in [random.randint(0, len(cv_images)-1) for i in range(5)]:
            
            cv_text_image = cv_images[i]
            cv_text_image = cv.resize(cv_text_image, (320, 280))
            cv_text_image = cv.cvtColor(cv_text_image, cv.COLOR_GRAY2RGB)
            cv_text_image = cv.putText(cv_text_image, str(categorielist_reduced[int(y_predicted[i])]), (10,250), font, 1, (0, 255, 0), 1, cv.LINE_AA)
            previewimgs.append(cv_text_image)
        
        preview = np.concatenate(previewimgs, axis=1)

        cv.imshow('preview', preview)
        cv.waitKey(0)
        cv.destroyAllWindows()

zip_file = "256_ObjectCategories.zip"


pname = sys.argv[0]
if len(sys.argv) == 4:
    #sift
    zip_file = sys.argv[1]
    save_file_additional = sys.argv[2]
    save_file_svm = sys.argv[3]
    evaluate(zip_file=zip_file, save_file_additional=save_file_additional, save_file_svm=save_file_svm)
else:
    print(pname+" <zip_file> <additional_data> <svm_model>")
