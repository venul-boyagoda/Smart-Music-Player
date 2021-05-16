from flask import Flask, render_template, Response
import cv2 as cv
from flask.helpers import url_for
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
import time
from collections import Counter
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

app = Flask(__name__)

#Load pre-trained CNN model and weights
json_file = open('cnn_model2.json', 'r')
cnn_model_json = json_file.read()
json_file.close()
cnn_model = model_from_json(cnn_model_json)
cnn_model.load_weights('cnn_model2.h5')

#Load face detection model from OpenCV
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Initialize webcam stream
cap = cv.VideoCapture(0)

#Initialize list of music genres
genres = ['pop', 'rap', 'rock', 'edm', 'latin', 'indie', 'clasical', 'country']

#Load environment variables
load_dotenv()

#Generator function that will stream video frames and results to client
def gen():
        while True:
            emotion_list = []
            reset_time = time.time() + 10 * 1
            img = None
            genre = random.choice(genres)

            while (time.time() < reset_time):
                _, img = cap.read()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = img[int(y):int(y+h), int(x):int(x+w)]
                    face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                    face = cv.resize(face, (48, 48))

                    pixels = image.img_to_array(face)
                    pixels = np.expand_dims(pixels, axis=0)
                    pixels = pixels/255

                    predictions = cnn_model.predict(pixels)

                    max_index = np.argmax(predictions[0])

                    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    emotion = emotions[max_index]

                    emotion_list.append(emotion)

                    cv.putText(img, emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, jpeg = cv.imencode('.jpg', img)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            counter = Counter(emotion_list)
            emotion_final, _ = counter.most_common(1)[0]

            if (emotion_final == 'angry' or emotion_final == 'disgust'):
                genre = random.choice([x for x in genres if x != genre])
            elif emotion_final == 'neutral':
                genre = random.choice(genres)
            elif (emotion_final == 'fear' or emotion_final == 'sad'):
                genre = 'blues'

            spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
            response = spotify.search(q='genre:' + genre, type='track', market='US', offset=0, limit=1)

            song = response['tracks']['items'][0]['name']
            artists = ''
            for i in response['tracks']['items'][0]['artists']:
                artists += i['name'] + ', '
            artists = artists[:-2]
            message = song + ' by ' + artists

            reset_time = time.time() + 20
            while (time.time() < reset_time):
                _, img = cap.read()   
                cv.putText(img, message, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                _, jpeg = cv.imencode('.jpg', img)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_emotions')
def live_emotions():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)