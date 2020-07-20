import RPi.GPIO as GPIO
import time
from threading import Thread

lockers_buttons_pins = {1: 37, 2: 38, 3: 40}


def check_lockers_buttons():
    while True:
        # try to get the lock
        try:
            GPIO.setmode(GPIO.BOARD)
            for key, value in lockers_buttons_pins.items():
                GPIO.setup(value, GPIO.IN)
                if GPIO.input(value):
                    print("Closing Locker " + str(key))

        finally:
            GPIO.cleanup()
            time.sleep(1)



buttons_thread = Thread(target=check_lockers_buttons, args=())
buttons_thread.start()