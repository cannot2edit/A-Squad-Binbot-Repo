import time
import numpy as np
from picamera2 import Picamera2
from PIL import Image
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
from gpiozero import Servo
from gpiozero import Button
from time import sleep

servo1 = Servo (
    18,
    min_pulse_width = 0.0014,
    max_pulse_width = 0.0016
    )

servo2 = Servo (
    12,
    min_pulse_width = 0.0014,
    max_pulse_width = 0.0016
    )

button1 = Button(3)
button2 = Button(13)

servo1.value = None
servo2.value = None

def OneEightyACW(servo):
    servo.value = 1
    sleep(0.5)
    servo.value = None
    
    
def WestQuadrant():
    OneEightyACW(servo1)
    servo2.value = 1
    
    def pressed():
        servo1.value = None
        servo2.value = None
        
    button1.when_pressed = pressed


with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
    
interpreter = tf.lite.Interpreter(model_path = "model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

picam2 = Picamera2()
config = picam2.create_preview_configuration (
    main = {"format": "RGB888", "size": (224,224)}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Camera Up")


try:
    while True:
        
        frame = picam2.capture_array()
        
        image = Image.fromarray(frame)
        
        input_data = np.expand_dims(image, axis = 0).astype(np.float32)
        input_data /= 255.0
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        index = np.argmax(output_data)
        confidence = output_data[index]
        
        print(f"Prediction: {labels[index]} (confidence: {confidence:.2f})")
        
        if labels[index] == "4 Cardboard":
            WestQuadrant()
        
        time.sleep(25)
        
except KeyboardInterrupt:
    print("Stopping camera")
    picam2.stop()