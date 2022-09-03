import operator
import re
import string

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer
from keras.models import model_from_json
import nltk
from nltk.data import load
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("tagsets")

# Declare a flask app
app = Flask(__name__)

specter_model = SentenceTransformer("allenai-specter")
bert_model = SentenceTransformer("bert-base-nli-mean-tokens")

json_file = open("spectermodel.json", "r")
spectermodel_json = json_file.read()
json_file.close()
specter_keras_model = model_from_json(spectermodel_json)
specter_keras_model.load_weights("spectermodel.h5")

json_file = open("bertmodel.json", "r")
bertmodel_json = json_file.read()
json_file.close()
bert_keras_model = model_from_json(bertmodel_json)
bert_keras_model.load_weights("bertmodel.h5")

json_file = open("spectermodel_classification.json", "r")
spectermodel_classification_json = json_file.read()
json_file.close()
specter_classification_keras_model = model_from_json(spectermodel_classification_json)
specter_classification_keras_model.load_weights("spectermodel_classification.h5")

json_file = open("bertmodel_classification.json", "r")
bertmodel_classification_json = json_file.read()
json_file.close()
bert_classification_keras_model = model_from_json(bertmodel_classification_json)
bert_classification_keras_model.load_weights("bertmodel_classification.h5")

json_file = open("posmodel.json", "r")
pos_json = json_file.read()
json_file.close()
pos_keras_model = model_from_json(pos_json)
pos_keras_model.load_weights("posmodel.h5")
possible_tags = list(load("help/tagsets/upenn_tagset.pickle").keys())

print('Go to http://127.0.0.1:5000/')


@app.route('/response', methods=['POST'])
def response():
    text = request.form.get("text")

    sentences = []
    specter_keywords = []
    bert_keywords = []
    specter_classification_keywords = []
    bert_classification_keywords = []
    pos_keywords = []

    mod_text = re.sub("\\((.*?)\\)", " ", text)
    mod_text = re.sub("\\[(.*?)\\]", " ", mod_text)
    mod_text = re.sub("\\s{2,}", " ", mod_text)

    for sentence in mod_text.split('.'):
        if sentence == "" or sentence.isspace() or sentence.strip().isnumeric() or sentence is None:
            continue

        # sentence = re.sub("\\((.*?)\\)", " ", sentence)
        # sentence = re.sub("\\[(.*?)\\]", " ", sentence)
        # sentence = re.sub("\\s{2,}", " ", sentence)
        sentences += [sentence]

        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).strip().lower()

        mod_sentence = re.sub(
            "|".join(np.append(np.char.add(np.char.add(" ", np.array(stopwords.words("english"))), " "), ", ")), " ",
            re.sub(" ", "  ", " " + sentence + " "))

        mod_sentence = re.sub("\\s{2,}", " ", mod_sentence).strip()

        specter_encoded = specter_model.encode(mod_sentence)
        bert_encoded = bert_model.encode(mod_sentence)

        specter_keywords += [
            mod_sentence.split(" ")[round(specter_keras_model.predict(np.array([specter_encoded]))[0][0])]]
        bert_keywords += [mod_sentence.split(" ")[round(bert_keras_model.predict(np.array([bert_encoded]))[0][0])]]
        specter_classification_keywords += [mod_sentence.split(" ")[min(np.argmax(
            specter_classification_keras_model.predict(np.array([specter_encoded]))[0]),
            len(mod_sentence.split(" ")) - 1)]]
        bert_classification_keywords += [mod_sentence.split(" ")[min(np.argmax(
            bert_classification_keras_model.predict(np.array([bert_encoded]))[0]), len(mod_sentence.split(" ")) - 1)]]

        if np.array(nltk.pos_tag(nltk.word_tokenize(sentence))).shape[0] == 0:
            continue

        tags = np.array(nltk.pos_tag(nltk.word_tokenize(sentence)))[:, 1]
        dummies = pd.get_dummies(tags)

        for i in possible_tags:
            if i not in dummies.columns:
                dummies[i] = 0

        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        dummies = dummies.to_numpy()
        dummies = np.vstack((dummies, np.zeros((200 - dummies.shape[0], len(possible_tags)))))

        pos_keywords += [sentence.split(" ")[min(np.argmax(pos_keras_model.predict(np.array([dummies]))[0]),
                                                 len(sentence.split(" ")) - 1)]]

    merged = ""
    keywords = []
    count = 1
    for i in zip(sentences, specter_keywords, bert_keywords, specter_classification_keywords,
                 bert_classification_keywords, pos_keywords):
        scores = dict.fromkeys(i[1:], 0)
        scores[i[1]] += 1
        scores[i[2]] += 2
        scores[i[3]] += 4.5
        scores[i[4]] += 4.5
        scores[i[5]] += 5
        keyword = max(scores, key=lambda key: scores[key])

        sentence_words = i[0].split(" ")
        sentence_words = [x for x in sentence_words if x]
        mod_sentence_words = i[0].translate(str.maketrans('', '', string.punctuation)).strip().split(" ")
        indices = [i for i, x in enumerate(mod_sentence_words) if x.lower() == keyword]

        for a in indices:
            keyword = sentence_words[a].translate(str.maketrans('', '', string.punctuation)).strip()
            sentence_words[a] = "____________"

        merged += str(count) + ". " + " ".join(sentence_words) + ".\n\n"
        keywords += [str(count) + ". " + keyword]
        count += 1

    return render_template("index.html", text=text, questions=merged[:len(merged) - 2], answers="\n".join(keywords))


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
