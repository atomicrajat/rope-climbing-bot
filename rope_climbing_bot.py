#Project: Rope Climbing Bot

#Import required libraries
from yolov5_tflite_inference import yolov5_tflite #For inference
import argparse #For command line arguments
import cv2 #For image capture and manipulation
from PIL import Image #For image proccessing
from utils import letterbox_image, scale_coords
import numpy as np
import time
import RPi.GPIO as GPIO    # Import Raspberry Pi GPIO library
from time import sleep     # Import the sleep function from the time module
from gpiozero import AngularServo

#Picamera2 for raspberry pi Bullseye (Comment this for Buster) 
#from picamera2.encoders import H264Encoder
#from picamera2 import Picamera2, Preview

#Picamera for raspberry pi Buster (comment This for Bullseye)
from picamera.array import PiRGBArray
from picamera import PiCamera


##############################################################################################

#Pin and Camera Initialization
GPIO.setwarnings(False)    # Ignore warning for now
GPIO.setmode(GPIO.BCM)   # Use physical pin numbering


#Initialize motor diver pins using RPi.GPIO (Here)
motor_pin1 = 23
motor_pin2 = 24
motor_speed_pin = 25
GPIO.setup(motor_pin1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(motor_pin2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(motor_speed_pin, GPIO.OUT)
speed_control = GPIO.PWM(motor_speed_pin,1000)

#Initialize servo motor for package dropping
servo_pin = 12 #its a PWM pin
servo = AngularServo(servo_pin, min_pulse_width=0.0006,max_pulse_width=0.0023)

#Example for setting led for demo
led_pin = 5
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)   # Set pin 5 to be an output pin and set initial value to low (off)

#Camera initialization
#input size
w = 640
h = 480
#Picamera2 (Bulseye)

#picam = Picamera2()
#picam.configure(picam.create_preview_configuration(main={"format": 'YUV420', "size": (640, 480)}))
#picam.start()

#Picamera (Buster)

picam = PiCamera()
picam.resolution = (w,h)
rawCapture = PiRGBArray(picam,size=(w,h))



##############################################################################################
#variables
zone_interval = 5 #indicates the seconds to reach next zone (change it accordingly)
capture_interval = 3 #indicates the seconds to scan the area and capture the frame (change accordingly)
load_interval = 10 #indicates the seconds to load the package back into servo
dropped = False
speed = 25 #speed of the motor (change accordingly [L:25, M:50, H:75])
zone = 1
target_zone = 3
img_size = 416
conf_thres = 0.70
iou_thres = 0.45
webcam = 0
weights = 'yolov5s-fp16.tflite'
path  = "forward"
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (20, 40)   
# fontScale 
fontScale = 0.5 
# Blue color in BGR 
color = (0, 255, 0)   
# Line thickness of 1 px 
thickness = 1
###############################################################################################

#Load the AI model
person_detection_model = yolov5_tflite(weights,img_size,conf_thres,iou_thres)
size = (img_size,img_size)


#Functions to capture and process the image

def capture_and_process_image(zone):
    global dropped
    print("Scanning the area")
    sleep(capture_interval)
    #Capture the frame in YUV420 Format and convert to RGB (Bullseye)
    #yuv420 = picam.capture_array()
    #frame = cv2.cvtColor(yuv420, cv2.COLOR_YUV420p2RGB)
    
    #Capture image in bgr and then convert into array for opencv (Buster)
    picam.capture(rawCapture,format="bgr")
    frame = rawCapture.array
    rawCapture.truncate(0)
    #resize the image
    image_resized = letterbox_image(Image.fromarray(frame),size)
    image_array = np.asarray(image_resized)
    #Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255.0
    
    #Person detection (Inference)
    result_boxes, result_scores, result_class_names = person_detection_model.detect(normalized_image_array)
    print("Processing done..")
    
    
    if "person" in result_class_names:
        #set the led Indicator HIGH
        GPIO.output(led_pin, GPIO.HIGH)
        
        #Print the info
        print(f"[ALERT] Total {result_class_names.count('person')} Persons Detected In Zone {zone}")
        
        #save the image with bbox and label
        for i,r in enumerate(result_boxes):
           if result_class_names[i] == "person":
               org = (int(r[0]),int(r[1]))             
               cv2.rectangle(image_array, (int(r[0]),int(r[1])), (int(r[2]),int(r[3])), (255,0,0), 1)
               cv2.putText(image_array, str(int(100*result_scores[i])) + '%  ' + str(result_class_names[i]), org, font,  
                           fontScale, color, thickness, cv2.LINE_AA)
           
        cv2.imwrite(f"Zone_{zone}_image.jpg",image_array)
        
        #send the message with image (Optional)
        #send_msg_with_image(frame)
        
        #trigger the servo and drop the package
        if not dropped:
            dropped = drop_package()
        else:
            print("[Package already dropped]")
        
        #set the led Indicator LOW
        GPIO.output(led_pin, GPIO.LOW)
    else:
        GPIO.output(led_pin, GPIO.LOW)
        print(f"No Person Detected In Zone {zone}")
        
# Function to move the bot forward       
def move_forward():
    GPIO.output(motor_pin1, GPIO.HIGH)
    GPIO.output(motor_pin2, GPIO.LOW)
    print("Bot is moving forward") 
 
# Function to move the bot backward
def move_backward():
    GPIO.output(motor_pin1, GPIO.LOW)
    GPIO.output(motor_pin2, GPIO.HIGH)
    print("Bot is moving backward")
    
# Function to stop the bot
def stop_bot():
    GPIO.output(motor_pin1, GPIO.LOW)
    GPIO.output(motor_pin2, GPIO.LOW)
    print("Bot is stopped") 

def drop_package():
    print("droping package..")
    servo.angle = 90
    return True      
    


########################################################################################################

#start the Bot
speed_control.start(speed) #set the speed of the motor before moving forward 
move_forward() #start the bot and move forward

########################################---Simulation---###########################################################    
try:
    while True:
        print("Moving to next zone")
        sleep(zone_interval)
        print(f"Zone {zone} Reached")
        stop_bot()
        capture_and_process_image(zone)
        if zone == target_zone:                      #If its last zone return to base 
            print("Reached last zone, returning..")
            if path == "forward":
                path = "backward"
                target_zone = 0
                servo.angle = 0 #change the servo angle back to 0 degree when the bot reaches back to zone 0 (initial place)
                dropped = False #setting the dropped variable back to False when it reaches initial place
                print("Loading the package")
                sleep(load_interval)
            elif path == "backward":
                path = "forward"
                target_zone = 3
        if path == "forward":
            move_forward()                         #else move forward to next zone if path is forward
            zone=zone+1
        elif path == "backward":
            move_backward()                        #move backward to next zone if path is backward
            zone=zone-1
        else:
            stop_bot()
except:
    stop_bot()
    print("ERROR")
        
######################################################################################################################        
            
