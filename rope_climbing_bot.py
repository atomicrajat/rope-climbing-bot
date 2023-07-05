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

#Picamera2 for raspberry pi Bullseye (Comment this for Buster) 
#from picamera2.encoders import H264Encoder
#from picamera2 import Picamera2, Preview

#Picamera for raspberry pi Buster (comment This for Bullseye)
from picamera.array import PiRGBArray
from picamera import PiCamera


##############################################################################################

#Pin and Camera Initialization
GPIO.setwarnings(False)    # Ignore warning for now
GPIO.setmode(GPIO.BOARD)   # Use physical pin numbering


#Initialize motor diver pins using RPi.GPIO (Here)
motor_pin1 = 23
motor_pin2 = 24
GPIO.setup(motor_pin1, GPIO.OUT)
GPIO.setup(motor_pin2, GPIO.OUT)

#Initialize servo motor for package dropping


#Initialize IR sensor for detecting the zone
ir_sensor_pin = 5
GPIO.setup(ir_sensor_pin, GPIO.IN)


#Example for setting led for demo
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)   # Set pin 8 to be an output pin and set initial value to low (off)

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
zone = 0
target_zone = 4
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
    print("Scanning the area")
    sleep(3)
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
        GPIO.output(8, GPIO.HIGH)
        
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
        
        
        #set the led Indicator LOW
        GPIO.output(8, GPIO.LOW)
    else:
        GPIO.output(8, GPIO.LOW)
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
    


########################################################################################################

#start the Bot
move_forward()

###################################################---With all sensors---########################################
#Check for zones
# try:
#     while True:
#         ir_sensor_pin_value = GPIO.input(ir_sensor1_pin) #read the ir sensor
        
#         if ir_sensor_pin_value == 0:                     #when black strip i.e Zone encountered
#             print(f"Zone {zone} Reached")
#             stop_bot()                                  #Stop the bot
#             capture_and_process_image(zone)            #Check for person
#             #If bot reaches the last zone
#             if zone == target_zone:                      #If its last zone return to base 
#                 print("Reached last zone, returning..")
#                 if path == "forward":
#                     path = "backward"
#                     target_zone = 0
#                 elif path == "backward":
#                     path = "forward"
#                     target_zone = 4

#             if path == "forward":
#                 move_forward()                         #else move forward to next zone if path is forward
#                 zone=zone+1
#             elif path == "backward":
#                 move_backward()                        #move backward to next zone if path is backward
#                 zone=zone-1
#             else:
#                 stop_bot()
#         else:
#             print("Moving to next zone...")
            
###################################################################################################################

########################################---Simulation---###########################################################    
try:
    while True:
        print("Moving to next zone")
        sleep(5)
        print(f"Zone {zone} Reached")
        stop_bot()
        capture_and_process_image(zone)
        if zone == target_zone:                      #If its last zone return to base 
            print("Reached last zone, returning..")
            if path == "forward":
                path = "backward"
                target_zone = 0
            elif path == "backward":
                path = "forward"
                target_zone = 4
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
            
