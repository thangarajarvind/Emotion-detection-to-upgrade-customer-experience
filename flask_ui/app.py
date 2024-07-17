from flask import Flask, render_template, request
import random
from unittest import result
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import string
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import v_a
from v_a import Video_and_audio
app = Flask(__name__, template_folder='template')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
#model = 0
model = keras.models.load_model(r"C:\Users\thang\Downloads\Sentiment\Code\tf_lstmmodel.h5")
counter=0

sentiment_dict = 0

def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' '
    #print(delete_dict)
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    #print(text1)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>2))]) 
    #print(text2)
    return text2.lower()

review_data = pd.read_csv(r"C:\Users\thang\Downloads\Sentiment\Code\test.csv")
review_data.dropna(axis = 0, how ='any',inplace=True) 
review_data['Review'] = review_data['Review'].apply(clean_text)
review_data['Review_Split'] = review_data['Review'].apply(lambda x:len(str(x).split())) 


lb = LabelEncoder() 
review_data['Label'] = lb.fit_transform(review_data['Label'])

mask = (review_data['Review_Split'] < 100) & (review_data['Review_Split'] >=20)
df_short_reviews = review_data[mask]
mask = review_data['Review_Split'] >= 100
df_long_reviews = review_data[mask]

X_train, X_valid, y_train, y_valid = train_test_split(review_data['Review'].tolist(),review_data['Label'].tolist(),test_size=0.5,stratify = review_data['Label'].tolist(),random_state=0)

num_words = 50000
tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
tokenizer.fit_on_texts(X_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/customer')
def customer():
   return render_template('customer.html',sentiment_dict=sentiment_dict,counter=counter,exec=0)

@app.route('/customer-result',methods = ['POST', 'GET'])
def result():
   global counter
   if request.method == 'POST':
      result = request.form.get('Enter text for Sentiment Analysis')

      #result = "This is the good batch"
      counter+=1

      def Sentiment(x):
        y=[]
        y.append(x)
        print(y)
        print(str(tokenizer.texts_to_sequences(['xyz how are you'])))
        x_tok = np.array(tokenizer.texts_to_sequences(y))
        print(x_tok)
        x_tok = pad_sequences(x_tok, padding='post', maxlen=100)
        predictions = model.predict(x_tok)
        print(predictions[0][0])
        if predictions[0][0] <0.3:
            return "Negative"
        elif predictions[0][0] >0.6:
            return "Positive"
        else:
            return "Neutral"

      sentiment_dict = Sentiment(result)

      #Database setup
      import mysql.connector

      mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="",
      database="emotion"
      )

      mycursor = mydb.cursor()

      #Getting total number of executives
      sql = "SELECT * FROM executive"
      mycursor.execute(sql)
      myresult = mycursor.fetchall()
      exec_count = 0
      for x in myresult:
        exec_count = exec_count+1

      set_size = exec_count//3
      
      #Positive Feedback
      if(sentiment_dict == 'Positive'):
          sql = "SELECT * FROM executive ORDER BY score LIMIT %s"
          val = (set_size,)
          mycursor.execute(sql, val)
          exec = mycursor.fetchall()
          idx = random.randint(0,set_size-1)
          print(exec[idx])

      if(sentiment_dict == 'Negative'):
          sql = "SELECT * FROM executive ORDER BY score DESC LIMIT %s"
          val = (set_size,)
          mycursor.execute(sql, val)
          exec = mycursor.fetchall()
          idx = random.randint(0,set_size-1)
          print(exec[idx])

      if(sentiment_dict == 'Neutral'):
          sql = "SELECT * FROM executive ORDER BY score DESC LIMIT %s,%s"
          val = (set_size,set_size,)
          mycursor.execute(sql, val)
          exec = mycursor.fetchall()
          idx = random.randint(0,set_size-1)
          print(exec[idx])

      return render_template('customer.html',sentiment_dict=sentiment_dict,counter=counter,exec=exec[idx])


      '''sid_obj = SentimentIntensityAnalyzer()
      sentiment_dict = sid_obj.polarity_scores(result)
      print(sentiment_dict)
      counter+=1
      sentiment_dict['text']=result
      if sentiment_dict['compound'] >= 0.05 :
          sentiment_dict['sentiment']="Positive"
      elif sentiment_dict['compound'] <= - 0.05 :
          sentiment_dict['sentiment']="Negative"
      else :
          sentiment_dict['sentiment']="Neutral"
      return render_template('customer.html',sentiment_dict=sentiment_dict,counter=counter)'''

@app.route('/service')
def service():
    score = 0
    return render_template('ccare.html',score=score,counter=counter)

@app.route('/service-result',methods = ['POST', 'GET'])
def service_result():
    global counter
    

    if request.method == 'POST':
        counter+=1
        eid = request.form.get('eid')
        cname = request.form.get('cname')

        video = request.files['video']
        audio = request.files['audio']
        video.save(os.path.join('Videos/temp.mp4'))
        audio.save(os.path.join('Audios/temp.wav'))
        score = Video_and_audio.score()

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
        val = (cname,)
        mycursor.execute(sql, val)
        print(mycursor.rowcount, "customer record inserted.")

        #Obtaining cid
        sql = "SELECT cid FROM customer ORDER BY cid DESC LIMIT 1"

        mycursor.execute(sql)

        myresult = mycursor.fetchall()

        for x in myresult:
            cid = x[0]

        #Call detail insertion
        sql = "INSERT INTO call_details (eid, cid, score) VALUES (%s, %s, %s)"
        val = (eid, cid, score,)
        mycursor.execute(sql, val)
            
        mydb.commit()

        print(mycursor.rowcount, "call detail record inserted.")

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

        return render_template('ccare.html',score=score,counter=counter)

if __name__ == '__main__':
   app.run(debug = True)