import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import colorsys
import random
#from test import *
from keras import backend as K



def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def preprocess_image(image, model_image_size=(300,300)):    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_AREA)
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image, 0)  # Add batch dimension.

    return image

def preprocess_image_for_tflite(image, model_image_size=300):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_image_size, model_image_size))
    image = np.expand_dims(image, axis=0)
    image = (2.0 / 255.0) * image - 1.0
    image = image.astype('float32')

    return image

def non_max_suppression(scores, boxes, classes, max_boxes=10, min_score_thresh=0.5):
    out_boxes = []
    out_scores = []
    out_classes = []
    if not max_boxes:
        max_boxes = boxes.shape[0]
    for i in range(min(max_boxes, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            out_boxes.append(boxes[i])
            out_scores.append(scores[i])
            out_classes.append(classes[i])

    out_boxes = np.array(out_boxes)
    out_scores = np.array(out_scores)
    out_classes = np.array(out_classes)

    return out_scores, out_boxes, out_classes

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    h, w, _ = image.shape
    name= ""
    detected_objects=[]

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        #name = '{} '.format(predicted_class)
        detected_objects.append(predicted_class)

        ###############################################
        # yolo
        #top, left, bottom, right = box
        ###############################################

        ###############################################
        # ssd_mobilenet
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = (xmin * w, xmax * w,
                                  ymin * h, ymax * h)
        ###############################################

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        #print(name)
        # print(label, (left, top), (right, bottom))
                
        # colors: RGB, opencv: BGR
        cv2.rectangle(image, (left, top), (right, bottom), tuple(reversed(colors[c])), 6)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom), tuple(reversed(colors[c])), -1)

        cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        #name = '{} '.format(predicted_class)
        
    return image,detected_objects

def get_voice():
    r = sr.Recognizer()
    
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Say something!")
            try:
                audio = r.listen(source)
            except sr.WaitTimeoutError:
                #speech("Timeout occurred. Please speak again.")
                continue

        try:
            command = r.recognize_google(audio)
            print("You said: ", command)
            return command
            # get_command()
        except Exception:
            pass


def speech(text):
    tts=gTTS(text=text,lang='en')
    filname="speech.mp3"
    tts.save(filname)
    playsound.playsound(filname)
    os.remove(filname)



conditions=["what do you see", "second service","what is it","tell me what do you see","detect it","see it","tell","tell me","detect",
            "what's in front of me","front" ,"in front"]
cond=["help","help me","hey siri","hey assistant","hello","hello assistant"]
cond2=["yes","yeah","yup"]
close_labels=["no","close","terminate"]

def get_command():
    while True:
        command = get_voice()
        if not command:
            # No speech input was detected, wait a bit and try again
            time.sleep(1)
            continue
        for condition in conditions:
            if condition in command:
                speech('Okay sir, I will detect it for you')
                real_time_object_detection_on_one_frame(interpreter, colors)
                break
                #return

        for condition in cond:
            if condition in command:
                speech('Hello sir , I can help you if you want to ask me what i see now , if You want this say "yes"')
                command=get_voice()
                for condition in cond2 :
                    if condition in command:
                        speech("okay i will detect it for you")
                        real_time_object_detection_on_one_frame(interpreter, colors)
                        break
                        #return
        for condition in close_labels:
            if condition in command:
                speech("Okay i will close it now")
                return
                
        # If no recognized command was found, try again
        time.sleep(3)
        speech('If you want to try again tell me waht do you want?')
        #speech('Sorry, I didn\'t understand. Please try again.')


def run_detection(image, interpreter, wanted_classes):

    # Run model: start to detect
    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]['index'], image)
    # Invoke the interpreter.
    interpreter.invoke()

    # get results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)

    # Filter out unwanted classes
    idxs = np.where(np.isin(classes, wanted_classes))
    boxes = boxes[idxs]
    scores = scores[idxs]
    classes = classes[idxs]

    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), 'images/dog.jpg'))

    return out_scores, out_boxes, out_classes




def real_time_object_detection_on_one_frame(interpreter, colors):
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read() 
    list_of_numbers = list(range(1,90))
    image_data = preprocess_image_for_tflite(frame, model_image_size=300)
    out_scores, out_boxes, out_classes = run_detection(image_data, interpreter,wanted_classes=list_of_numbers)
    # Draw bounding boxes on the image file
    result,objects= draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
    cv2.imshow("Object detection - ssdlite_mobilenet_v2", frame)
    cv2.waitKey(1)
    camera.release   
    time.sleep(2)
    cv2.destroyAllWindows()
    if len(objects)==0:
        speech("there is no objects to detect")
    elif len(objects)==1:
        for object in objects:
            speech("the object is " +object)
    else:
        All_objects="and".join(objects)
        speech("The objects is ")
        speech(All_objects)



if __name__ == '__main__':
    #Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="ssdlite_mobilenet_v2.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # label
    class_names = read_classes('coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    #face_Recognition()
    #real_time_object_detection_on_one_frame(interpreter,colors)
    get_command()

