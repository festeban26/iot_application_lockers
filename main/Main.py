from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub, SubscribeListener
import uuid
import random
import threading
from threading import Thread
import RPi.GPIO as GPIO
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
# import OpenCV module
import cv2
# import os module for reading training data directories and paths
import os
import sys
# import numpy to convert python lists to numpy arrays as
# it is needed by OpenCV face recognizers
import numpy as np
import warnings

lockers_owners_dic = {'Meli': 1, 'Majo': 1, 'Esteban': 1, 'Pau': 2, 'Pala': 2, 'Ana': 2, 'Mare': 3, 'Noe': 3}
subjects = ["", "Meli", "Pau", "Mare", "Noe", "Majo", "Pala", "Esteban", "Ana"]
lockers_lock_status_is_open = {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True,
                               9: True, 10: True, 11: True}
lockers_servos_pins = {1: 7, 2: 11, 3: 12, 4: 13, 5: 15, 6: 16, 7: 18, 8: 22, 9: 29, 10: 31, 11: 32}
lockers_servos_open_position = {1: 3, 2: 0, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 15, 9: 15, 10: 15, 11: 15}
lockers_servos_closed_position = {1: 70, 2: 73, 3: 80, 4: 70, 5: 70, 6: 70, 7: 70, 8: 70, 9: 70, 10: 70, 11: 70}
lockers_buttons_pins = {1: 37, 2: 38, 3: 40}

# Get the current path
current_path = os.getcwd()
training_data_path = os.path.join(current_path, "data", "training-data")
# load cascade classifier training file for LBP cascade
lbp_cascade = cv2.CascadeClassifier(os.path.join(current_path, "opencv", "sources", "data",
                                                 "lbpcascades", "lbpcascade_frontalface_improved.xml"))
# sets the face recognition classifier
face_recognition_classifier = lbp_cascade
# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
consecutive_times_to_accept_face_rec_prediction = 3
min_confidence_to_match = 100

pn_publish_key = 'INSERT API KEY'
pn_subscribe_key = 'INSERT API KEY'


def check_lockers_buttons(pn_instance, pn_instance_channel):
    while True:
        for key, value in lockers_buttons_pins.items():
            GPIO.setup(value, GPIO.IN)
            if GPIO.input(value):
                print("Closing Locker " + str(key) + " on button pressed command")
                set_locker_lock_state(key, False, pn_instance, pn_instance_channel)

        time.sleep(1)


def set_servo_angle(servo_pin, angle):
    duty = angle / 18 + 2  # min 2, max 12
    GPIO.setup(servo_pin, GPIO.OUT)
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(7.5)
    try:
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.8)
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)
    finally:
        pwm.stop()


def set_locker_lock_state(locker_number, to_open, pn_instance, pn_instance_channel):
    lock.acquire()
    servo_pin = lockers_servos_pins[locker_number]
    if to_open:
        if lockers_lock_status_is_open[locker_number] is False:
            set_servo_angle(servo_pin, lockers_servos_open_position[locker_number])

            for key, value in lockers_owners_dic.items():
                if value == locker_number:
                    response = ['Server', key, 'open', pn_instance.uuid]
                    pn_instance.publish().channel(pn_instance_channel).message(response).sync()

            lockers_lock_status_is_open[locker_number] = True
            print("Locker (" + str(locker_number) + ") opened")

    else:
        if lockers_lock_status_is_open[locker_number] is True:
            set_servo_angle(servo_pin, lockers_servos_closed_position[locker_number])

            for key, value in lockers_owners_dic.items():
                if value == locker_number:
                    response = ['Server', key, 'closed', pn_instance.uuid]
                    pn_instance.publish().channel(pn_instance_channel).message(response).sync()

            lockers_lock_status_is_open[locker_number] = False
            print("Locker (" + str(locker_number) + ") closed")
    lock.release()


def start_pub_nub_server(pn_instance, pn_instance_channel):
    print("Initializing Pub Nub Server")
    try:
        my_listener = SubscribeListener()
        pn_instance.add_listener(my_listener)
        pn_instance.subscribe().channels(pn_instance_channel).execute()
        my_listener.wait_for_connect()
        while True:
            result = my_listener.wait_for_message_on(pn_instance_channel)
            if result.message[1] == 'Server':  # if it is for me
                print("Just received a message: ", end='')
                print(result.message)
                petitioner = result.message[0]
                if petitioner in lockers_owners_dic:
                    if result.message[2] == 'locker_status':
                        locker_status = lockers_lock_status_is_open[lockers_owners_dic[petitioner]]
                        response = ['Server', petitioner]
                        if locker_status:
                            response.append("open")
                        else:
                            response.append("closed")
                        response.append(pn_instance.uuid)
                        print("Responding: ", end='')
                        print(response)
                        pn_instance.publish().channel(pn_instance_channel).message(response).sync()
                    elif result.message[2] == 'open':
                        # Call function to open the locker
                        print("Opening " + petitioner + "'s locker")
                        thread_to_open_locker = Thread(target=set_locker_lock_state,
                                                       args=(lockers_owners_dic[petitioner],
                                                             True,
                                                             pn_instance,
                                                             pn_instance_channel,))
                        thread_to_open_locker.start()
                    elif result.message[2] == 'close':
                        # Call function to close the locker
                        print("Closing " + petitioner + "'s locker")
                        thread_to_open_locker = Thread(target=set_locker_lock_state,
                                                       args=(lockers_owners_dic[petitioner],
                                                             False,
                                                             pn_instance,
                                                             pn_instance_channel,))
                        thread_to_open_locker.start()

    finally:
        pn_instance.unsubscribe().channels(pn_instance_channel).execute()
        print("PubNub instance unsubscribed")
    return


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
            print("[MIN: " + str(min_confidence_to_match) + "] Face found: ", end='')
            print(label)
            if label[1] <= min_confidence:
                # get name of respective label returned by face recognizer
                label_text = subjects[label[0]]
                detected_subjects_indexes.append(label[0])
                # draw a rectangle around face detected
                draw_rectangle(img, detected_coordinates[detected_faces_counter])
                # draw name of predicted person
                draw_text(img, label_text, detected_coordinates[detected_faces_counter][0],
                          detected_coordinates[detected_faces_counter][1] - 5)
    return img, detected_subjects_indexes


def main(pn_instance, pn_channel):
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
        print("Camera sensor is setting its light levels")
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
                    subject_name = subjects[subject]
                    subject_locker = lockers_owners_dic[subject_name]
                    thread = Thread(target=set_locker_lock_state,
                                    args=(subject_locker,
                                          True,
                                          pn_instance,
                                          pn_channel,))
                    thread.start()
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


try:
    GPIO.setmode(GPIO.BOARD)
    lock = threading.Lock()
    pn_config = PNConfiguration()
    pn_config.publish_key = pn_publish_key
    pn_config.subscribe_key = pn_subscribe_key
    pn_config.uuid = get_uuid_from_default_uuid_file()
    # instantiate a PubNub instance
    pn = PubNub(pn_config)
    pn_channel = 'iot-lockers'
    pub_nub_server_thread = Thread(target=start_pub_nub_server, args=(pn, pn_channel,))
    pub_nub_server_thread.start()

    buttons_thread = Thread(target=check_lockers_buttons, args=(pn, pn_channel,))
    buttons_thread.start()
    main(pn, pn_channel)

except KeyboardInterrupt:
    print('Interrupted')
    sys.exit(0)
finally:
    GPIO.cleanup()
