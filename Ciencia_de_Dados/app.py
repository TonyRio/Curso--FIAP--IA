from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
#model = pickle.load(open("Heart.pkl"))

@app.route("/", methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restcg = int(request.form['restcg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak= float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        values = np.array([[age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]])
       # prediction = model.predict(values)

        return render_template('result.html')

    if __name__ =="__main__":
        app.run(debug=True)
