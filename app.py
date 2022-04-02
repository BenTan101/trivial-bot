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

print('Go to http://127.0.0.1:5000/')


@app.route('/response', methods=['POST'])
def response():
    model_name = request.form['action']
    text = request.form.get("text")

    if model_name == 'compare all!':
        sentences = []
        specter_keywords = []
        bert_keywords = []
        specter_classification_keywords = []
        bert_classification_keywords = []

        for sentence in text.split('.'):
            if sentence == "" or sentence.isspace() or sentence.strip().isnumeric():
                continue

            sentence = re.sub("\\((.*?)\\) ", " ", sentence).lower()
            sentence = re.sub("\\[(.*?)\\] ", " ", sentence).lower()
            sentence = re.sub("[-–—§!\"#$%&'()*+./:;<=>?@[\\]^_`{|}~,‘’…]+", "", sentence).strip()

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

            specter_encoded = specter_model.encode(mod_sentence)
            bert_encoded = bert_model.encode(mod_sentence)

            sentences += [sentence]
            specter_keywords += [mod_sentence.split(" ")[
                                     round(specter_keras_model.predict(np.array([specter_encoded]))[0][0])
                                 ]]
            bert_keywords += [mod_sentence.split(" ")[
                                  round(bert_keras_model.predict(np.array([bert_encoded]))[0][0])
                              ]]
            specter_classification_keywords += [mod_sentence.split(" ")[
                                                    min(np.argmax(specter_classification_keras_model.predict(
                                                        np.array([specter_encoded]))[0]),
                                                        len(mod_sentence.split(" ")) - 1,
                                                        )
                                                ]]
            bert_classification_keywords += [mod_sentence.split(" ")[
                                                 min(np.argmax(
                                                     bert_classification_keras_model.predict(np.array([bert_encoded]))[
                                                         0]),
                                                     len(mod_sentence.split(" ")) - 1,
                                                 )
                                             ]]

        merged = ""
        for i in zip(sentences, specter_keywords, bert_keywords, specter_classification_keywords,
                     bert_classification_keywords):
            merged += "sentence: " + i[0] + "\nspecter keyword: " + i[1] + "\nbert keyword: " + i[
                2] + "\nspecter (classification) keyword: " + i[3] + "\nbert (classification) keyword: " + i[
                          4] + "\n\n\n"
    else:
        is_classification = 'classification' in model_name

        if 'allenai-specter' in model_name:
            model = specter_model
            keras_model = specter_classification_keras_model if is_classification else specter_keras_model
        elif 'bert-base-nli-mean-tokens' in model_name:
            model = bert_model
            keras_model = bert_classification_keras_model if is_classification else bert_keras_model

        sentences = []
        keywords = []

        for sentence in text.split('.'):
            if sentence == "" or sentence.isspace() or sentence.strip().isnumeric():
                continue

            sentence = re.sub("\\((.*?)\\) ", " ", sentence).lower()
            sentence = re.sub("\\[(.*?)\\] ", " ", sentence).lower()
            sentence = re.sub("[-–—§!\"#$%&'()*+./:;<=>?@[\\]^_`{|}~,‘’…]+", "", sentence).strip()

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
                             min(np.argmax(keras_model.predict(np.array([encoded]))[0]),
                                 len(mod_sentence.split(" ")) - 1,
                                 )
                         ]] if is_classification else [
                mod_sentence.split(" ")[round(keras_model.predict(np.array([encoded]))[0][0])]]

        merged = ""
        for i in zip(sentences, keywords):
            merged += "question: " + i[0].replace(" " + i[1] + " ", " " + "_" * 10 + " ") + \
                      "\nkeyword: " + i[1] + "\n\n\n"

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
