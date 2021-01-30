from flask import Flask, render_template, request
import tensorflow
from pickle5 import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize, sent_tokenize
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords
import numpy as np
from src.database import nlpDB
import os

app = Flask(__name__)

# path
path = os.getcwd()

# Loading of tokenizer pickle file
# tokenizer = pickle.load(open(path + "/essay_auto_grading/src/tokenizer.pkl", "rb"))
tokenizer = pickle.load(open("../essay_auto_grading/src/tokenizer.pkl", "rb"))

# Load .h5 file
# model = load_model(path + '/essay_auto_grading/src/functional_model.h5')
model = load_model('../essay_auto_grading/src/functional_model.h5')

# Dababase
db = nlpDB()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            essay = request.form["essay"]
            essay_set = int(request.form["essay_set"])

            # tokensisation
            essay_token = tokenizer.texts_to_sequences([essay])
            essay_token_pad = pad_sequences(essay_token, maxlen=500)

            # word count
            # Set of stopwords from nltk.corpus package
            stop = set(stopwords.words('english'))
            word_count = len([i for i in word_tokenize(essay) if i not in stop])

            # sentence count
            sent_count = len(sent_tokenize(essay))

            # stacking of (essay_set, sent_count, word_count)
            features = np.column_stack((essay_set, sent_count, word_count))

            # Prediction
            result = np.argmax(model.predict([essay_token_pad, features]))

            # uploading data in db
            try:
                db.updateDataBase(essay, int(essay_set), int(result))

            except Exception as e:
                print("Database Error: ", e)


            return render_template("home.html", prediction = "Your score is {}".format(result))

        except Exception as e:
            print("Something went wrong: ", e)


    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
