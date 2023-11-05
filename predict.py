from flask import Flask
import pickle
from flask import request
from flask import jsonify


model = 'midterm_model'

with open(model,'rb')as f_in:
    dv,model = pickle.load(f_in)

app = Flask('heartdisease')
@app.route('/predict',methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    heartdisease = y_pred>=0.5

    result = {'heart_disease_probability': float(y_pred),
              'heart_disease':bool(heartdisease)}
    return  jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)