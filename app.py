import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

model1 = pickle.load(open('models/model.pkl', 'rb'))


@app.route('/')
def home ():
    return render_template("index.html")
  
  
@app.route('/predict', methods =['POST'])
def predict():
    print("this is the shit")
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    
    prediction = model1.predict(features)
    
   
    return render_template ("index.html", prediction_text = 'Possibility for diabetes is $ {}'.format(prediction))
  
  
if __name__ == "__main__":
    app.run(port=3000)