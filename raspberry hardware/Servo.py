import threading
from threading import Thread

import RPi.GPIO as GPIO
import time

lockers_servos_pins = {1: 3, 2: 5, 3: 7}
lockers_servos_open_position = {1: 3, 2: 0, 3: 15}
lockers_servos_closed_position = {1: 70, 2: 73, 3: 80}
lockers_lock_status_is_open = {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True,
                               9: True, 10: True, 11: True}


def set_servo_angle(servo_pin, angle):
    duty = angle / 18 + 2
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(servo_pin, GPIO.OUT)
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(7.5)
    # min 2, max 12
    try:
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.8)
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)
    finally:
        pwm.stop()
        GPIO.cleanup()


def set_locker_lock_state(locker_number, to_open, threading_lock):
    threading_lock.acquire()
    servo_pin = lockers_servos_pins[locker_number]
    if to_open:
        lockers_lock_status_is_open[locker_number] = True
        set_servo_angle(servo_pin, lockers_servos_open_position[locker_number])
    else:
        set_servo_angle(servo_pin, lockers_servos_closed_position[locker_number])
        lockers_lock_status_is_open[locker_number] = False
    threading_lock.release()


lock = threading.Lock()
th1 = Thread(target=set_locker_lock_state, args=(1, False, lock))
th2 = Thread(target=set_locker_lock_state, args=(1, True, lock))
th3 = Thread(target=set_locker_lock_state, args=(1, False, lock))
th4 = Thread(target=set_locker_lock_state, args=(1, True, lock))
th1.start()
th2.start()
th3.start()
th4.start()