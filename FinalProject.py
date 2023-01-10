#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import face_recognition as fr
import cv2
import os
from datetime import datetime
from tkinter import *   
  
top = Tk()  
top.title("Auto-Attendance")
top['bg']= 'blue'
top.geometry("222x100")  

def markAttendance(name):
    with open('/Users/apple/Desktop/AttendanceList - Sheet1.csv','r+') as FILE:
        allLines = FILE.readlines()
        AttendanceList = []
        for line in allLines:
            entry = line.split(',')
            AttendanceList.append(entry[0])
        if name not in AttendanceList:
            now = datetime.now()
            dtString = now.strftime('%d/%b/%Y, %H:%M:%S')
            FILE.writelines(f'\n{name},{dtString}')

def detection():  
#Read two known faces
    face_1 = fr.load_image_file("Lebron.jpeg")
    face_2 = fr.load_image_file("MJ.jpeg")
    face_3 = fr.load_image_file("pic2_.png")

#Generate ecoding of two faces
    face_1_encoding = fr.face_encodings(face_1)[0]
    face_2_encoding = fr.face_encodings(face_2)[0]
    face_3_encoding = fr.face_encodings(face_3)[0]

    known_encodings = [face_1_encoding, face_2_encoding, face_3_encoding]
    known_names = ['LeBron James', 'Michael Jordan', 'Arhaan Goyal']

    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        face_locations = fr.face_locations(frame)
        face_encodings = fr.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = fr.compare_faces(known_encodings, face_encoding)

            namee = "Unkown "

            face_distances = fr.face_distance(known_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                namee = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, namee, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            markAttendance(namee)
            cv2.imshow('Webcam_facerecognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video_capture.release()

    cv2.destroyAllWindows() 
  
  
b2 = Button(top,text = "Take attendance",command = detection,activeforeground = "blue",activebackground = "pink",pady=20) 
b2.place(relx=0.5, rely=0.5, anchor=CENTER)



top.mainloop()  


# In[ ]:




