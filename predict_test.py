import requests

url = 'http://localhost:9696/predict'

patient_id = 'patient-0001'
patient = {'Age': 41,
  'Sex': 'M',
  'ChestPainType': 'ASY',
  'RestingBP': 110,
  'Cholesterol': 289,
  'FastingBS': 0,
  'RestingECG': 'Normal',
  'MaxHR': 170,
  'ExerciseAngina': 'N',
  'Oldpeak': 0,
  'ST_Slope': 'Flat',
  }

response = requests.post(url,json=patient).json()
print(response)

if response['heart_disease']== True:
    print(f'{patient_id} has heartdisease')
else:
    print(f'{patient_id} is healthy')