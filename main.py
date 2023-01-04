from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from scaling import return_scalar

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_dia():
    if request.method == "POST":
        preg = float(request.form["pregnancies"])
        glu = float(request.form["glucose"])
        blood = float(request.form["blood"])
        skin = float(request.form["skin"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dfs = float(request.form["dpf"])
        age = float(request.form["age"])

        scaler = return_scalar()
        arr = np.array([preg, glu, blood, skin, insulin, bmi, dfs, age]).reshape(1,-1)
        arr = scaler.transform(arr)
        result = model.predict(arr)

        if (result[0] == 0):
            return render_template('index.html',result="Is not Diabetic")
        else:
            return render_template('index.html',result="Is Diabetic")


if __name__ == "__main__":
    app.run(debug=True)
