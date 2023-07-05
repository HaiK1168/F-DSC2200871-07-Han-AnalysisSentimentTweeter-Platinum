import re
import pandas as pd
import numpy as np
import sqlite3
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

app = Flask(__name__)

with open('model/model_mlp_kombinasi.pickle','rb') as f:
   modelMLP = pickle.load(f)

with open('model/Vectorize_kombinasi.pickle','rb') as f:
   Vectorize = pickle.load(f)

with open('model/Transformer_kombinasi.pickle','rb') as f:
   Transformer = pickle.load(f)

# Melakukan preprocessing

kamus_alay = pd.read_csv('Data/new_kamusalay.csv', encoding='Latin-1',names=('alay','baku'))

# Melakukan Cleaning tweet tanpa menghapus stopwords
def text_proses(text):

    # Untuk stemmer kata
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    if type(text) == np.float64:
        return ""
    temp = re.sub(r'\b(USER|RT|URL)\b', '', text)
    temp = temp.lower()
    temp = re.sub(r"@[A-Za-z0-9_]+","", temp) #Menghilangkan Mention
    temp = re.sub("#[A-Za-z0-9_]+","", temp) #Menghilangkan Hashtag
    temp = re.sub(r'https://\S+', '', temp)
    # Melakukan subtitusi kata alay dengan kata baku
    temps = temp.split()
    for i in range(len(temps)):
        temp = temps[i]
        replacement = kamus_alay[kamus_alay['alay'] == temp]['baku']
        if not replacement.empty:
            temps[i] = replacement.iloc[0]
    temp = " ".join(temp.strip() for temp in temps)
    # temp = " ".join(temp.strip() for temp in temps if temp not in stopwords)

    # Mengubah angka menjadi terbaca
    # temps = temp.split()
    # for i in range(len(temps)):
    #     temp = temps[i]
    #     if temps[i].isdigit() == True :
    #         temps[i] = Terbilang().parse(temps[i]).getresult()
    #     else :
    #         temps[i]
    # temp = " ".join(temps)

    # Melakukan stemming
    tokens = word_tokenize(temp)
    temp = ' '.join([word for word in tokens ])
    temp = stemmer.stem(temp)

    temp = re.sub(r'\\x..', '', temp) #Menghilangkan Emoticon
    temp = re.sub(r'&amp;', 'dan', temp)
    temp = re.sub(r'http\S+', '', temp) #Menghilangkan link
    temp = re.sub(r"www.\S+", "", temp) #Menghilangkan link
    temp = re.sub(r"\\[n|t]", "", temp) #Menghilangkan 'enter' dan 'tab'
    temp = re.sub("[^A-Za-z\s']","", temp) #Menghilangkan yang bukan huruf
    temps = temp.split()
    temp = " ".join(temp.strip() for temp in temps)

    return temp


app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host = LazyString(lambda: request.host)
    
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)



# Bagian File Processing LSTM
@swag_from("docs/file_lstm.yml", methods=['POST'])
@app.route('/file-processing-LSTM', methods=['POST'])

def text_processing_file_LSTM():

    file = open('model/LSTM/tokenizer.pickle','rb')
    tokenizer = pickle.load(file)
    file.close()

    sentiment = ['negative', 'neutral', 'positive']

    model_lstm = load_model('model/LSTM/LSTM2106.h5')
    file = open('model/LSTM/X_pad_sequances.pickle','rb')
    feature_lstm = pickle.load(file)
    file.close()

    # Upladed file
    file = request.files.getlist('file')[0]
    
    # Import file csv ke Pandas
    input_df = pd.read_csv(file)
    assert input_df.columns == 'text'

    results = []
    for _, row in input_df.iterrows():
        input_text = row['text'] 

        processed_text = text_proses(input_text)

        feature = tokenizer.texts_to_sequences(processed_text)
        feature = pad_sequences(feature, maxlen = feature_lstm.shape[1])

        prediction = model_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        result = {
            'Sentence': processed_text,
            'Sentiment': get_sentiment
        }
        results.append(result)

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': results, 
    }

    response_data = jsonify(json_response)
    return response_data


# Bagian Text Processing LSTM
@swag_from("docs/text_lstm.yml", methods=['POST'])
@app.route('/text-lstm', methods=['POST'])

def text_processing_lstm():
    file = open('model/LSTM/tokenizer.pickle','rb')
    tokenizer = pickle.load(file)
    file.close()

    sentiment = ['negative', 'neutral', 'positive']

    model_lstm = load_model('model/LSTM/LSTM2106.h5')
    file = open('model/LSTM/X_pad_sequances.pickle','rb')
    feature_lstm = pickle.load(file)
    file.close()

    ori_text = request.form.get('text')
    input_text = [text_proses(ori_text)]

    feature = tokenizer.texts_to_sequences(input_text)
    feature = pad_sequences(feature, maxlen = feature_lstm.shape[1])

    prediction = model_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {"Sentence" : input_text,
                 'Sentiment': get_sentiment}, 
    }

    response_data = jsonify(json_response)
    return response_data


# Bagian File Processing MLP
@swag_from("docs/file_processing_NN.yml", methods=['POST'])
@app.route('/file-processing-NN', methods=['POST'])

def text_processing_file_MLP():

    # Upladed file
    file = request.files.getlist('file')[0]
    
    # Import file csv ke Pandas
    input_df = pd.read_csv(file)
    assert input_df.columns == 'text'

    results = []
    for _, row in input_df.iterrows():
        input_text = row['text'] 

        processed_text = text_proses(input_text)
        text = Vectorize.transform([processed_text])
        text = Transformer.transform(text)
        pred = modelMLP.predict(text)[0]

        result = {
            'Sentence': processed_text,
            'Sentiment': pred
        }
        results.append(result)

    json_response = {
        'status_code': 200,
        'description': "File yang sudah diproses",
        'data': results, 
    }

    response_data = jsonify(json_response)
    return response_data


# Bagian Text Processing MLP
@swag_from("docs/text_processing_NN.yml", methods=['POST'])
@app.route('/text-processing-NN', methods=['POST'])

def text_processing_MLP():

    original_text = request.form.get('text')
    input_text = text_proses(original_text)
    text = Vectorize.transform([input_text])
    text = Transformer.transform(text)

    pred = modelMLP.predict(text)[0]

    json_response = {
        'status_code': 200,
        'description': "Social Status description",
        'data': {"Sentence" : input_text,
                 'Sentiment': pred}, 
    }

    response_data = jsonify(json_response)
    return response_data



if __name__ == '__main__':
   app.run(debug=True)