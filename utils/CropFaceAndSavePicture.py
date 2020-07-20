# import OpenCV module
import cv2
# import os module for reading training data directories and paths
import os
# matplotlib for display our images
import matplotlib.pyplot as plt

current_path = os.getcwd()

# load cascade classifier training file for haarcascade
lbp_cascade = cv2.CascadeClassifier(os.path.join(current_path, "opencv", "sources", "data",
                                                       "lbpcascades", "lbpcascade_frontalface_improved.xml"))

face_recognition_classifier = lbp_cascade


def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# function to detect face using OpenCV
def detect_face(f_classifier, img, filename, scale_factor=1.1):

    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images

    faces = f_classifier.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=3)
    # go over list of faces and draw them as rectangles on original colored
    for (x, y, w, h) in faces:
        cv2.imwrite(str(filename + 62) + ".png", img[y-40:y+h+40, x-40:x+w+40])




# Variables that will be used to walk the training data path
training_data_path, training_data_dirs, training_data_files \
    = next(os.walk(os.path.join(current_path, "data", "training-data")))
for idx_i, val_i in enumerate(training_data_dirs):
    # Get the sub folders paths
    sub_folder_path = os.path.join(training_data_path, val_i)
    # Variables that will be used to walk the sub folders paths
    path, dirs, files = next(os.walk(sub_folder_path))
    # Each sub folder file
    for idx_j, val_j in enumerate(files):
        full_file_path = os.path.join(path, val_j)
        # load image
        test = cv2.imread(full_file_path)
        # call our function to detect faces
        detect_face(face_recognition_classifier, test, idx_j)
