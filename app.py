# Flask
# Some utilites
import re

import numpy as np
from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer
from keras.models import model_from_json
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# TODO: jk just reference https://analyticsarora.com/how-to-make-a-flask-web-app-for-keras-model/
# Compile to exe: https://elc.github.io/posts/executable-flask-pyinstaller/

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

print('Go to http://127.0.0.1:5000/')


@app.route('/response', methods=['POST'])
def response():
    model_name = request.form['action']
    if model_name == 'allenai-specter':
        model = specter_model
        keras_model = specter_keras_model
    else:
        model = bert_model
        keras_model = bert_keras_model

    text = request.form.get("text")
    sentences = []
    keywords = []

    for sentence in text.split('.'):
        if sentence == '':
            continue

        sentence = re.sub("\\((.*?)\\) ", " ", sentence).lower()
        sentence = re.sub("\\[(.*?)\\] ", " ", sentence).lower()
        sentence = re.sub("[-–—§!\"#$%&'()*+./:;<=>?@[\\]^_`{|}~,‘’…]+", "", sentence)

        mod_sentence = re.sub(
            "|".join(
                np.append(
                    np.char.add(
                        np.char.add(" ", np.array(stopwords.words("english"))), " "
                    ),
                    ", ",
                )
            ),
            " ",
            re.sub(" ", "  ", " " + sentence + " "),
        )

        mod_sentence = re.sub("\\s{2,}", " ", mod_sentence).strip()
        encoded = model.encode(mod_sentence)

        sentences += [sentence]
        keywords += [mod_sentence.split(" ")[
                         round(keras_model.predict(np.array([encoded]))[0][0])
                     ]]

    merged = ""
    for i in zip(sentences, keywords):
        merged += "sentence: " + i[0] + "\nkeyword: " + i[1] + "\n\n\n"

    return render_template("index.html", text=text, model=model_name, keywords=merged)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
