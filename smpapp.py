from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('student_marks_predictor_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('smp.html')

@app.route('/predict',methods=['POST'])  
def predict():
   
    input_features =[float(x) for x in request.form.values()]
    features_value =np.array(input_features)
    output =model.predict([features_value])[0][0].round(2)  
    return render_template('smp.html',predict_text='if you study for {}  hours per day then you will get [{}%] marks in exam'.format(input_features, output))

if __name__ == '__main__':
    app.run(debug=True)