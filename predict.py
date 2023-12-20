from flask import Flask
import pickle
from flask import request
from flask import jsonify


model = 'model_capstone'

with open(model,'rb')as f_in:
    dv,model = pickle.load(f_in)

app = Flask('drug_classification')
@app.route('/predict',methods=['POST'])
def predict():
    test_1 = request.get_json()
    X = dv.transform([test_1])
    y_pred = model.predict_proba(X)[0, 1]

    result = {'input':test_1,
              'output':y_pred}
    return  jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)