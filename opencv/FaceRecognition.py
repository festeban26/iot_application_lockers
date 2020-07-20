
from picamera.array import PiRGBArray
from picamera import PiCamera
# import OpenCV module
import cv2
# import os module for reading training data directories and paths
import os
import sys
import time
# import numpy to convert python lists to numpy arrays as
# it is needed by OpenCV face recognizers
import numpy as np
import warnings

import uuid
import random

# Get the current path
current_path = os.getcwd()
# load cascade classifier training file for LBP cascade
lbp_cascade = cv2.CascadeClassifier(os.path.join(current_path, "opencv", "sources", "data",
                                                 "lbpcascades", "lbpcascade_frontalface_improved.xml"))
# sets the face recognition classifier
face_recognition_classifier = lbp_cascade
# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# subjects list
subjects = ["", "Meli", "Pau", "Mariemi", "Noe", "Majo", "Ana Paula", "Esteban"]
training_data_path = os.path.join(current_path, "data", "training-data")

consecutive_times_to_accept_face_rec_prediction = 3
min_confidence_to_match = 115


def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# function to draw rectangle on image according to given (x, y) coordinates and given width and height
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


# function to detect faces using OpenCV
def detect_faces(img, scale_factor=1.1):
    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # let's detect multiscale (some images may be closer to camera than others) images
    detected_faces_coordinates = \
        face_recognition_classifier.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=3)
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


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    # list to hold all subject faces
    training_data_subjects_faces = []
    # list to hold labels for all subjects
    training_data_subjects_labels = []
    # let's go through each directory and read images within it
    for dir_name in dirs:
        # our subject directories start with letter 's' so ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # ------STEP-2--------
        # extract label number of subject from dir_name format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        subject_label = int(dir_name.replace("s", ""))
        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image, detect face and add face to list of faces
        for image_name in subject_images_names:
            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue
            # build image path. sample image path = training-data/s1/1.pgm
            image_path = os.path.join(subject_dir_path, image_name)
            # read image
            subject_image = cv2.imread(image_path)
            # detect faces
            subject_detected_faces, subject_detected_faces_coordinates = detect_faces(subject_image)

            # ------STEP-4--------
            # if more than one face is detected on the picture, issue a warning. The traing-data images
            # should only contain one face per picture
            if len(subject_detected_faces) is 1:
                # we will ignore faces that are not detected. Also, on the training process
                if subject_detected_faces[0] is not None:
                    # add face to list of faces
                    training_data_subjects_faces.append(subject_detected_faces[0])
                    # add label for this face
                    training_data_subjects_labels.append(subject_label)
            else:
                warnings.warn("More than one face detected on a training data image: " + image_path, UserWarning)
    return training_data_subjects_faces, training_data_subjects_labels


# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img, min_confidence=100):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    detected_faces, detected_coordinates = detect_faces(img)
    detected_subjects_indexes = []
    if detected_faces is not None:
        for detected_faces_counter, detected_face in enumerate(detected_faces):
            # predict the image using our face recognizer
            label = face_recognizer.predict(detected_face)
            # TODO
            print(label)
            if label[1] <= min_confidence:
                # get name of respective label returned by face recognizer
                label_text = subjects[label[0]]
                detected_subjects_indexes.append(label[0])
                # draw a rectangle around face detected
                draw_rectangle(img, detected_coordinates[detected_faces_counter])
                # draw name of predicted person
                draw_text(img, label_text, detected_coordinates[detected_faces_counter][0], detected_coordinates[detected_faces_counter][1] - 5)
    return img, detected_subjects_indexes


def get_random_uuid_as_string():
    basename_string = 'festeban26-iot-lockers-server'
    new_uuid_random_number = random.random()
    new_uuid_string = basename_string + "-" + str(new_uuid_random_number)
    custom_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, new_uuid_string)
    print("UUID generated: " + str(custom_uuid))
    return str(custom_uuid)


def get_uuid_from_default_uuid_file():
    # first check if a previous file exists, if it does not exist, generate uuid and save it to file
    try:
        with open("UUID.txt", "r") as text_file:
            return text_file.read()
    except IOError:
        print("UUID file was not found. Creating new uuid")
        new_uuid = get_random_uuid_as_string()
        save_uuid_in_default_uuid_file(new_uuid)
        with open("UUID.txt", "r") as text_file:
            return text_file.read()


def save_uuid_in_default_uuid_file(uuid_as_string):
    # first check if it exists, if it exists, delete the file
    try:
        file = open("UUID.txt", 'r')
        file.close()
        os.remove("UUID.txt")
    except IOError:
        pass
    # once the file is deleted or the file did not exists, save content
    with open("UUID.txt", "w") as text_file:
        text_file.write(uuid_as_string)


def main():
    # let's first prepare our training data. Data will be in two lists of same size
    # one list will contain all the faces and other list will contain respective labels for each face
    print("Preparing training data...")
    faces, labels = prepare_training_data(training_data_path)
    print("Data prepared")
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    print("Training model...")
    # train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
    print("Model trained")

    print("Initializing camera...")
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    try:
        camera.resolution = (352, 352)
        camera.framerate = 8
        time.sleep(15)  # give the sensor time to set its light levels.
        rawCapture = PiRGBArray(camera, size=(352, 352))
        print("Camera is ready")

        print("Predicting images...")
        face_recognition_effectiveness_dic = {}
        for counter, subject in enumerate(subjects):
            face_recognition_effectiveness_dic[counter] = 0
        effectiveness_last_list = []
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array
            predicted_img, subjects_in_img = predict(image, min_confidence_to_match)

            reset_effectiveness_counter_elements = list(set(effectiveness_last_list) - set(subjects_in_img))
            for element in reset_effectiveness_counter_elements:
                face_recognition_effectiveness_dic[element] = 0

            for subject in subjects_in_img:
                face_recognition_effectiveness_dic[subject] += 1
                if face_recognition_effectiveness_dic[subject] >= consecutive_times_to_accept_face_rec_prediction:
                    print(subject)
                    # call function to open locker
                    face_recognition_effectiveness_dic[subject] = 0
            effectiveness_last_list = subjects_in_img

            cv2.imshow("Face Recognition", predicted_img)  # show the frame
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  # clear the stream in preparation for the next frame
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                camera.close()
                break
    except KeyboardInterrupt:
        print('Interrupted')
        camera.close()
        sys.exit(0)



pub_num_iot_lockers_channel = 'iot-lockers'
pn_config = PNConfiguration()
pn_config.subscribe_key = 'INSERT API KEY'
pn_config.publish_key = 'INSERT API KEY'
pn_config.uuid = '4205e440-1258-462c-aa94-1870f1ab4755'
pub_nub = PubNub(pn_config)

print(pub_nub.uuid)
pub_nub.publish().channel(pub_num_iot_lockers_channel).message("Test").sync()
'''

try:
    main()
except KeyboardInterrupt:
    print('Interrupted')
    sys.exit(0)
