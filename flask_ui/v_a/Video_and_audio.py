#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import math


# In[3]:


#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result


# In[4]:


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm','happy','angry']


# In[10]:


#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\thang\\Downloads\\Emotion\\Emotion\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        print(file,file_name)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# In[11]:


#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)


# In[12]:


x_test


# In[13]:


#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))


# In[14]:


#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# In[15]:


model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[16]:


model.fit(x_train,y_train)


# In[17]:


y_pred=model.predict(x_test)


# In[18]:



accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))


# In[19]:


# from IPython.display import Javascript
# from google.colab import output
# from base64 import b64decode

# RECORD = """
# const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
# const b2text = blob => new Promise(resolve => {
#   const reader = new FileReader()
#   reader.onloadend = e => resolve(e.srcElement.result)
#   reader.readAsDataURL(blob)
# })
# var record = time => new Promise(async resolve => {
#   stream = await navigator.mediaDevices.getUserMedia({ audio: true })
#   recorder = new MediaRecorder(stream)
#   chunks = []
#   recorder.ondataavailable = e => chunks.push(e.data)
#   recorder.start()
#   await sleep(time)
#   recorder.onstop = async ()=>{
#     blob = new Blob(chunks)
#     text = await b2text(blob)
#     resolve(text)
#   }
#   recorder.stop()
# })
# """

# def record(sec=3):
#   display(Javascript(RECORD))
#   s = output.eval_js('record(%d)' % (sec*1000))
#   b = b64decode(s.split(',')[1])
#   with open('audio.flac','wb') as f:
#     f.write(b)
#   return 'audio.flac'  # or webm ?


# In[20]:


# record()


# In[21]:


# pip install pydub


# In[23]:


face_classifier = cv2.CascadeClassifier(r'C:\Users\thang\Downloads\Emotion\Emotion\Video\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\thang\Downloads\Emotion\Emotion\Video\model.h5')

emotion_labels = ['Angry','Tensed','Neutral','Happy','Neutral', 'Tensed', 'Neutral']


# In[24]:


import wave
import contextlib
def dur(audpath):
    with contextlib.closing(wave.open(audpath,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


# In[27]:


from pathlib import PurePath
from pydub import AudioSegment

def score():
    id='1-2'
    angry=0
    tensed=0
    happy=0
    total=0
    #audpath='C:\\Users\\thang\\Downloads\\Emotion\\Emotion\\Video\\Video Dataset\\Audio\\WAV\\'+id+'.wav'
    #file_path = PurePath(audpath)
    #vidpath='C:\\Users\\thang\\Downloads\\Emotion\\Emotion\\Video\\Video Dataset\\'+id+'.mp4'
    flac_tmp_audio_data = AudioSegment.from_file(os.path.join('Audios/temp.wav'))
    flac_tmp_audio_data = flac_tmp_audio_data.set_channels(1)
    flac_tmp_audio_data.export("test.wav", format="wav")
    feature=extract_feature('test.wav', mfcc=True, chroma=True, mel=True)
    x_new=feature
    y_new=model.predict([x_new])
    print(y_new[0])
    audioscore=5
    if y_new[0]=='calm':
        audioscore=5
    if y_new[0]=='happy':
        audioscore=10
    if y_new[0]=='angry':
        audioscore=1
    cap = cv2.VideoCapture(os.path.join('Videos/temp.mp4'))
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
    print("Angry =",(angry/total)*100)
    print("Happy =",(happy/total)*100)
    print("Tensed =",(tensed/total)*100)
    videoscore=5-(angry/total*10)+(happy/total*10)-(tensed/2/total*10)
    score=(audioscore*40+videoscore*60)/100
    print("Score:",score)
    cap.release()
    cv2.destroyAllWindows()
    return score

# In[58]:
'''

#Update Customer name and Executive id

eid = "362"
customer_name = "alpha"


# In[76]:


#Database Update

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="emotion"
)

mycursor = mydb.cursor()

#Customer info insertion
sql = "INSERT INTO customer (name) VALUES (%s)"
val = (customer_name,)
mycursor.execute(sql, val)
print(mycursor.rowcount, "customer record inserted.")

#Obtaining cid
sql = "SELECT cid FROM customer ORDER BY cid DESC LIMIT 1"

mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
    cid = x[0]


# In[77]:


#Call detail insertion
sql = "INSERT INTO call_details (eid, cid, score) VALUES (%s, %s, %s)"
val = (eid, cid, score,)
mycursor.execute(sql, val)
    
mydb.commit()

print(mycursor.rowcount, "call detail record inserted.")


# In[78]:


#Obtaining previous score of executive
sql = "SELECT score FROM executive WHERE eid = %s"
val = (eid,)
mycursor.execute(sql, val)

myresult = mycursor.fetchall()

previous_score = 0

for x in myresult:
    previous_score = x[0]

print("Previous score of executive:")
print(previous_score)

mydb.commit()

#Getting total calls of the executive
sql = "SELECT * FROM call_details WHERE eid = %s"
val = (eid,)
mycursor.execute(sql, val)

tot_call = len(list(mycursor))

#Calculating new score and updating it back
new_score = ((previous_score * tot_call + score)/(tot_call + 1))
sql = "UPDATE executive SET score = %s WHERE eid = %s"
val = (new_score,eid,)

mycursor.execute(sql,val)
mydb.commit()

print("Updated Score of the executive:")
print(new_score)


# In[75]:


import questionary

questionary.text("What's your first name").ask()


# In[84]:


customer_name = input("Kindly provide your name for record purposes:")
query1 = input("Hello "+customer_name+" ,What could we help you with?")
query2 = input("Any other particular details you would like to add upon this issue?")
query3 = input("A customer care executive is being allocated for your request, meanwhile you could also state any other help you might need here:")


# In[86]:


from chatterbot import chatbot
from chatterbot.trainers import ListTrainer
 
#creating a new chatbot
chatbot = Chatbot('Edureka')
trainer = ListTrainer(chatbot)
trainer.train([ 'hi, can I help you find a course', 'sure Id love to find you a course', 'your course have been selected'])
 
#getting a response from the chatbot
response = chatbot.get_response("I want a course")
print(response)


# In[6]:


import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="emotion"
)

mycursor = mydb.cursor()

sql = "INSERT INTO executive (eid, name, score) VALUES (%s, %s, %s)"
val = ("065", "Barns", "1.956",)
mycursor.execute(sql, val)
    
mydb.commit()


# In[61]:


import random

sql = "SELECT * FROM executive ORDER BY score DESC LIMIT %s,%s"
val = (3,3,)
mycursor.execute(sql, val)
myresult = mycursor.fetchall()

#for x in myresult:
#   cid = x[0]
#   print(cid)
idx = random.randint(0,2)
print(myresult[idx])


# In[65]:


import random

sql = "SELECT * FROM executive ORDER BY score LIMIT %s"
val = (3,)
mycursor.execute(sql, val)
myresult = mycursor.fetchall()

for x in myresult:
    cid = x[0]
    print(cid)
#idx = random.randint(0,2)
#print(myresult[idx])

'''