#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


from deepface import DeepFace


# In[3]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[4]:


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


# In[5]:


while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,
               result['dominant_emotion'],
               (50,50),
               font, 3,
               (0, 0, 255),
               2,
               cv2.LINE_4)
    cv2.imshow('Original video', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


# In[ ]:


cap.release()


# In[17]:


cv2.destroyAllWindows()


# In[ ]:




