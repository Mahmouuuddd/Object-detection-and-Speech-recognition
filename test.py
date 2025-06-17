import cv2
import face_recognition
import os
import glob
import speech_recognition as sr
from gtts import gTTS
import time
import playsound
import pyttsx3





cap = cv2.VideoCapture(0)

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

rate = engine.getProperty("rate")
engine.setProperty('rate', 150)



engine.runAndWait()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()



def get_voice():
    r = sr.Recognizer()
    
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Say something!")
            try:
                audio = r.listen(source, timeout=3)
            except sr.WaitTimeoutError:
                talk("Timeout occurred. Please speak again.")
                continue

        try:
            command = r.recognize_google(audio)
            print("You said: ", command)
            return command
            # get_command()
            # get_command2()
        except Exception:
            pass



def speech(text):
    tts=gTTS(text=text,lang='ar')
    filname="speech.mp3"
    tts.save(filname)
    playsound.playsound(filname)
    os.remove(filname)

def release_video():
    cap.release()
    cv2.destroyAllWindows()

def IterationN():
    i = 0
    cv2.imshow('Face Recognition', frame)
    time.sleep(2)
    for i in range(150):
        if cv2.waitKey(1) == 13 or i == 50: #13 is the Enter Key
            break
    cv2.destroyAllWindows()
    cap.release()


cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def face_extractor(frame):
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y+h+50, x:x+w+50]

    return cropped_face


def Make_Folder():
    global name_of_folder
    speech("من فضلك اعطني اسمك")
    engine.runAndWait()
    name_of_folder = get_voice()
    parent_dir = "D:\\projects\\Graduation project\\Computer vision part\\Face recognition\\registered"
    path_of_newdir = os.path.join(parent_dir, name_of_folder)
    the_final_path = os.mkdir(path_of_newdir)
    the_final_dir = "D:\\projects\\Graduation project\\Computer vision part\\Face recognition\\registered\\{}".format(str(name_of_folder))
    global the_final_Dir
    the_final_Dir = os.path.join(the_final_dir, name_of_folder) 


known_faces = []
known_names = []
known_faces_paths = []


registered_faces_path = 'D:\\projects\\Graduation project\\Computer vision part\\Face recognition\\registered\\'


for name in os.listdir(registered_faces_path):
    images_mask = '%s%s\\*.jpg' % (registered_faces_path, name)
    images_paths = glob.glob(images_mask) 
    known_faces_paths += images_paths
    known_names += [name for x in images_paths]




def get_encodings(img_path):
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    return encoding[0]

known_faces = [get_encodings(img_path) for img_path in known_faces_paths]



def face_Recognition():
    count = 0
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(frame_rgb)
        for face in faces: # top, right, bottom, left
            top, right, bottom, left = face
            face_code = face_recognition.face_encodings(frame_rgb, [face])[0]

            results = face_recognition.compare_faces(known_faces, face_code, tolerance=0.6)
            if any(results):
                name = known_names[results.index(True)]
                speech(f"أهلا  {name}")
                #engine.runAndWait()
                IterationN()
            else:
                Make_Folder()
                while True:
                    ret, frame = cap.read()
                    if face_extractor(frame) is not False:
                        count += 1
                        face = cv2.resize(face_extractor(frame), (600, 600))
                                            
                        file_name_path = the_final_Dir  +" "+str(count) + '.jpg'
                        
                        cv2.imshow('Face Cropper', face)
                        cv2.imwrite(file_name_path, face)
                        
                    else:
                         print("Face not found")
                         pass

                    if cv2.waitKey(1) == 13 or count == 5: #13 is the Enter Key
                        break
                
   
                release_video()
                speech(f"أهلا صديقي الجديد {name_of_folder}")
                engine.runAndWait()


