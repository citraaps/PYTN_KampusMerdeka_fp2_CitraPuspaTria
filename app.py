import numpy as np
import pickle
from flask import Flask, render_template, request

model = pickle.load(open('model/model_svm.pkl','rb'))

app = Flask(__name__, template_folder="template")

@app.route("/")
def main():
    return(render_template('main.html'))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = {0: "tidak hujan", 1: "hujan"}

    return render_template('main.html', prediction_text='Jika hari ini {} dengan curah hujan sebesar {}mm, kecepatan angin sebesar {}km/h, kelembaban sebesar {}%, dan tekanan udara sebesar {}hpa, maka diprediksi esok hari akan {}'.format(features[4], features[0], features[1], features[2], features[3], output[prediction[0]]))

if __name__ == '__main__':
    app.run(debug=True)