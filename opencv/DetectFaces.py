# import OpenCV module
import cv2
# import os module for reading training data directories and paths
import os

# Get the current path
current_path = os.getcwd()

# load cascade classifier training file for LBP cascade
lbp_cascade = cv2.CascadeClassifier(os.path.join(current_path, "opencv", "sources", "data",
                                                 "lbpcascades", "lbpcascade_frontalface_improved.xml"))
# sets the face recognition classifier
face_recognition_classifier = lbp_cascade

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# function to detect faces using OpenCV
def detect_faces(img, scale_factor=1.1):

    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    detected_faces_coordinates = \
        face_recognition_classifier.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=4)

    print(len(detected_faces_coordinates))

    # if no faces are detected then return original img
    if len(detected_faces_coordinates) == 0:
        return None, None

    detected_faces = []
    # extract the faces areas
    for counter, detected_face in enumerate(detected_faces_coordinates):
        (x, y, w, h) = detected_faces_coordinates[counter]
        detected_faces.append(gray_img[y:y + w, x:x + h])

    # return only the face part of the image
    return detected_faces, detected_faces_coordinates

test_img = cv2.imread(os.path.join(current_path, "data", "test", "1.png"))
faces, coordinates = detect_faces(test_img)

print(coordinates)
for face in faces:
    cv2.imshow("Test", face)
    cv2.waitKey(0)

cv2.destroyAllWindows()