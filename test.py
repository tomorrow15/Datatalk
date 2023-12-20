import requests

url = 'http://localhost:9696/predict'


test_id = 'test-1'
#test_1 = df_test.to_dict(orient='records')


test_1 = {'sex': 'm',
  'bp': 'normal',
  'cholesterol': 'normal',
  'age_cal': '60',
  'chemical_cal': '10-20'}


response = requests.post(url,json=test_1).json()
print(response)

#X = dv.transform([test_1])
#y_pred = rf.predict_proba(X)[0, 1]


#print('input:',test_1)
#print('output:', y_pred)
