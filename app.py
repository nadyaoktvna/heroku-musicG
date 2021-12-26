import numpy as np
from flask import Flask, render_template, request
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            audio, sr = librosa.load(file)
            audio, _ = librosa.effects.trim(audio)
            audio = audio[:661500]

            mfccs_features = librosa.feature.mfcc(y = audio, sr = 22050, n_mfcc = 40)
            mfccs_mean = np.mean(mfccs_features.T, axis=0)
            mfccs_var = np.var(mfccs_features.T, axis=0)
            mfccs_scaled_features = mfccs_mean+mfccs_var
            mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)

            predicted = model.predict(mfccs_scaled_features)
            predicted_label = np.argmax(predicted, axis=1)

            if predicted_label == [[0]]:
                genre_detected = 'Blues'
            elif predicted_label == [[1]]:
                genre_detected = 'Classical'
            elif predicted_label == [[2]]:
                genre_detected = 'Country'
            elif predicted_label == [[3]]:
                genre_detected = 'Disco'
            elif predicted_label == [[4]]:
                genre_detected = 'Hip-hop'
            elif predicted_label == [[5]]:
                genre_detected = 'Jazz'
            elif predicted_label == [[6]]:
                genre_detected = 'Metal'
            elif predicted_label == [[7]]:
                genre_detected = 'Pop'
            elif predicted_label == [[8]]:
                genre_detected = 'Reggae'
            else:
                genre_detected = 'Rock'

    return render_template('index.html', prediction = genre_detected)


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000, debug=True)