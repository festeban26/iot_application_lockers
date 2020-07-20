from picamera.array import PiRGBArray
from picamera import PiCamera
from fractions import Fraction
import cv2
import time

print("Initializing camera...")

with PiCamera() as camera:
    try:
        camera.resolution = (352, 352)
        camera.framerate = Fraction(1, 6)
        camera.sensor_mode = 3
        camera.shutter_speed = 6000000
        camera.iso = 800
        time.sleep(30)  # give the sensor time to set its light levels.
        camera.exposure_mode = 'off'
        camera.framerate = 10
        rawCapture = PiRGBArray(camera, size=(352, 352))
        print("Camera is ready")
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array
            cv2.imshow("Test", image)  # show the frame
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  # clear the stream in preparation for the next frame
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                camera.close()
                break
    finally:
        camera.close()



