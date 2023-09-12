import cv2 as cv
import dataloader
import utilities
import datasaver
import extractors
import svm
import numpy as np
import tqdm



def prep(zip_file):
    save_data = {}

    print("get_filelist_from_zip")
    filelist = dataloader.get_filelist_from_zip(zip_file)
    filelist = [k for k in filelist if '.JPG' in k]
    
    print("extract_all_categories_from_path")
    categorielist, filelist = dataloader.extract_all_categories_from_path(filelist, 1)
    print(f'number of images: {len(filelist)}')

    categorielist_reduced = utilities.select_random_elements(categorielist, max_elements=len(set(categorielist)), random=False)

    print("categorielist:", categorielist_reduced)

    save_data["categorielist"] = categorielist_reduced

    print("limit_list_by_categories")
    categorielist, filelist = dataloader.limit_by_categories(categorielist,filelist, categorielist_reduced)

    print("label_encoding")
    label_encoding = utilities.label_encoding(categorielist, categorielist_reduced)

    print("import_images_from_zip")
    return dataloader.import_images_from_zip(zip_file, filelist, cv.IMREAD_GRAYSCALE), save_data, label_encoding

def train(cv_images, save_data, label_encoding, save_file_svm, save_file_additional, extract="shi-tomasi", maxCorners=160, qualityLevel=0.2, minDistance=3, blockSize=15, winSize = (32, 32), blockStride = (2,2), cellSize = (4,4), nbins = 6):
    if extract == "shi-tomasi":
        print("extract_descriptors_shi-tomasi")
        images_descriptors, cv_images = extractors.extract_descriptors_shi_tomasi(cv_images, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)
    elif extract == "hog":
        print("extract_descriptors_hog")
        images_descriptors, cv_images = extractors.extract_fvs_hog(cv_images, winSize = winSize, blockSize = blockSize, blockStride = blockStride, cellSize = cellSize, nbins = nbins)
        images_descriptors = np.squeeze(images_descriptors)
    else:
        print("extract", extract, "not supported")
        return
    save_data["extract"] = extract

    print("calc_cluster_centers")
    descriptors_center = extractors.calc_cluster_centers(utilities.flatten(images_descriptors), len(set(label_encoding)))
    save_data["descriptors_center"] = descriptors_center

    print("calc_bow")
    features = []
    for desc_num in tqdm.trange(len(images_descriptors)):
        features.append(extractors.calc_bow(images_descriptors[desc_num], descriptors_center))


    print("train_test_split_data")
    X_train_split, X_test_split, y_train_split, y_test_split = utilities.train_test_split_data(features, label_encoding)

    print("len(train):", len(X_train_split), "len(test):", len(X_test_split))


    svm_net = svm.create_svm(4, 1000, 1)
    svm_net = svm.train_svm(svm_net, X_train_split, y_train_split)
    y_predicted = svm.predict_multiple(svm_net, X_test_split)
    utilities.print_prediction(y_predicted, y_test_split)
    utilities.show_confusion_mat(y_test_split, y_predicted)


    

    
    # trainsplit
    # best_kernel = None
    # best_c = 0
    # best_g = 0
    # best_percentage = 0

    # #k = cv.ml.SVM_RBF
    # for c in [1000, 10000]:
    #     for k in [2,3,4,5]:
    #         for g in [1]:
    #             svm_gs = svm.create_svm(k, c, g)
    #             svm_gs = svm.train_svm(svm_gs, X_train_split, y_train_split)
    #             y_predicted = svm.predict_multiple(svm_gs, X_test_split)
    #             per = utilities.print_prediction(y_predicted, y_test_split)
    #             if per > best_percentage:
    #                 best_kernel = k
    #                 best_percentage = per
    #                 best_c = c
    #                 best_g = g
    #             print(extract + ", accuracy: "+ str(per)+", kernel: "+str(k)+", C: "+str(c)+", gamma: "+str(g)+"\n")
    #             f = open("gridsearch3.csv", "a")
    #             f.write(extract + "; "+ str(per)+"; "+str(k)+"; "+str(c)+"; "+str(g)+"; "+str(maxCorners)+ "; " + str(qualityLevel) +"; "+ str(minDistance)+"; "+ str(blockSize)+"; "+ str(winSize)+"; "+ str(blockStride)+"; "+ str(cellSize)+"; "+ str(nbins)+"\n")
    #             f.close()

    # print(f'Best overall combination is kernel {best_kernel} gamma {best_g} c {best_c} with {best_percentage}')


    if save_file_svm is not None:
        print("save svm in file")
        svm_net.save(save_file_svm)
    if save_file_additional is not None:
        print("save additional data in file")
        datasaver.save_data(save_data, save_file_additional)


# zip_file = "/home/sebastian/uibk/ss2023/computerVision/assignments/assignment4/data/leafs.zip"
zip_file = "/home/sebastian/uibk/ss2023/computerVision/assignments/assignment4/data/A Database of Leaf Images Practice towards Plant Conservation with Plant Pathology.zip"


save_file_additional_hog = "trained/hog_additional_data.dat"
save_file_svm_hog = "trained/hog_svm_model.xml"
save_file_additional_shi = "trained/shi_additional_data.dat"
save_file_svm_shi = "trained/shi_svm_model.xml"
cv_imgs, save_dict, label_enc = prep(zip_file=zip_file)


#shi-tomasi/sift
#in dataloader change resize to (1024, 1024)
train(cv_imgs, save_dict, label_enc, save_file_svm_shi, save_file_additional_shi, extract="shi-tomasi", maxCorners=900,  qualityLevel=0.00001, minDistance=2)
#hog
#in dataloader change resize to (256, 256)
train(cv_imgs, save_dict, label_enc, save_file_svm_hog, save_file_additional_hog, 
      extract="hog", winSize=(128, 128), blockSize=(16,16), blockStride=(4,4), cellSize=(8,8), nbins=9)