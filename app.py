from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
# from flask_restful import reqparse
import h5py
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
@app.route("/", methods=['GET'])
def hello():
    return "hey"
@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('LSTMGLOVE.hdf5', compile=False)
    MAX_NB_WORDS = 10000
    MAX_SEQUENCE_LENGTH = 100

    tokenizer  = Tokenizer(num_words = MAX_NB_WORDS)

    # lr = joblib.load("model.pkl")
    if model:
        try:
            json = request.get_json()  
            model_columns = ["neutral", "Magnifying/minimizing", "Personalization", "overgeneralization", "should statements"]
            print(json)
            # tst = [[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,
            #           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            #           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,
            #           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            #           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            #           0,   0,   0,   0,   0,   0,   0,   1,  40, 118]]
            tmp=(json['arr'])

            tokenizer.fit_on_texts(tmp)
            tmp =  tokenizer.texts_to_sequences(tmp)
            test_data = pad_sequences(tmp, maxlen=MAX_SEQUENCE_LENGTH)

            print(test_data)
            # vals=np.array(temp)
            # print(temp)
            # vals = np.expand_dims(vals, axis=0)
            prediction = model.predict(test_data)
            print("here:",prediction)        
            return jsonify({'prediction': str(prediction[0])})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')
if __name__ == '__main__':
    app.run(debug=True)