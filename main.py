import pickle
import pandas as pd


def classify_turkish(data):
    model_filename = 'finalized_model_tr.sav'
    model = pickle.load(open(model_filename, 'rb'))

    vectorizer_filename = 'finalized_vectorizer_tr.sav'
    tf = pickle.load(open(vectorizer_filename, 'rb'))

    text_matrix = tf.transform(data["text"])
    y = model.predict(text_matrix)
    data["label"] = y

    return data


def classify_english(data):
    model_filename = 'finalized_model_en.sav'
    model = pickle.load(open(model_filename, 'rb'))

    vectorizer_filename = 'finalized_vectorizer_en.sav'
    tf = pickle.load(open(vectorizer_filename, 'rb'))

    text_matrix = tf.transform(data["text"])
    y = model.predict(text_matrix)
    data["label"] = y

    return data


data = pd.read_csv("processed_tweets.csv", encoding="utf-8", index_col="id")
en_data = data[data["language"] == "en"]
tr_data = data[data["language"] == "tr"]

en_data = classify_english(en_data)
tr_data = classify_turkish(tr_data)

data = pd.concat([en_data, tr_data])
data.to_csv("labeled_tweets.csv", encoding="utf-8")
