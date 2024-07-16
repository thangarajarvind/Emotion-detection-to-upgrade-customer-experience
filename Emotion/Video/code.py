from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Abisheck Kathirvel\Downloads\Emotion-20220405T090602Z-001\Emotion\Video\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Abisheck Kathirvel\Downloads\Emotion-20220405T090602Z-001\Emotion\Video\model.h5')

emotion_labels = ['Angry','Tensed','Neutral','Happy','Neutral', 'Tensed', 'Neutral']

id='3-2'
path='C:\\Users\\Abisheck Kathirvel\\Downloads\\Emotion-20220405T090602Z-001\\Emotion\\Video\\Video Dataset\\'+id+'.mp4'
cap = cv2.VideoCapture(path)

angry=0
tensed=0
happy=0
total=0


while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    
    
    
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
    
                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                total+=1
                if label=='Angry':
                    angry+=1
                if label=='Happy':
                    happy+=1
                if label=='Tensed':
                    tensed+=1
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("Angry =",angry/total*100)
print("Happy =",happy)
print("Tensed =",tensed)
cap.release()
cv2.destroyAllWindows()