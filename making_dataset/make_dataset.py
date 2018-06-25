# -*- coding: utf-8 -*-

# 라즈베리 파이의 파이카메라를 활용하여 이미지 데이터셋 생성

import os
# PI camera Setting
import RPi.GPIO as GPIO
from time import sleep
from picamera import PiCamera

dir = ''

def Get_lastnum(dir):
    list = os.listdir(dir) # dir is your directory path
    number_files = len(list)
    #print(number_files)
    return number_files

camera = PiCamera()
Ready = False

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN)
def pick():
    print("Wait")
    camera.brightness = 50
    num = Get_lastnum(dir)
    camera.capture('dataset/'+ str(num) +'.jpg')
    print(str(num) + ".jpg is captured!")

# Main Code Start
print("Press the button")
try:
    while True:
        if GPIO.input(18) == False and Ready == False:  # Door Close
            print("Wait to open")
            sleep(1)
            Ready = True
        if GPIO.input(18) == True and Ready == True:  # Door Open
            while True:
                print("Door Open, Wait Close..")
                sleep(1)
                if GPIO.input(18) == False:
                    print("Door is Closed")
                    pick()
                    Ready = False
                    break
except KeyboardInterrupt:
    GPIO.cleanup()