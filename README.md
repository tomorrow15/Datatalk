# Midterm project ML Zoomcamp 2023: Heart Failure Prediction

# Introduction 

Heart failure is a serious medical condition where the heart is unable to pump blood effectively, leading to various health complications. It is a major global health problem, with an estimated 620 million people living with heart diseases worldwide. Each year, around 60 million people develop a heart disease. Early prediction of heart disease can significantly improve patient care and outcomes.

In this project, I used multiple machine learning models to predict heart disease, including logistic regression, decision tree, random forest, and XGBoost gradients. I selected the model with the best AUC score and deployed it in a Docker container.

# Dataset

This is heart prediction project that uses Heart Failure Prediction dataset from Kaggle datasets.  Here is the link to it: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction.

# Feature Information

Age: age of the patient 
Sex: sex of the patient 
ChestPainType: chest pain type 
RestingBP: resting blood pressure 
Cholesterol: serum cholesterol
FastingBS: fasting blood sugar
RestingECG: resting electrocardiogram results
MaxHR: maximum heart rate 
ExerciseAngina: exercise-induced angina 
Oldpeak: oldpeak 
ST_Slope: the slope of the peak exercise 
HeartDisease: output class 

# File Description

Dockerfile : heartdisease prediction service  including python image - python:3.11-slim,pipenv and hosting the server with waitress 
Pipfile: it describe which packages are installed in pipenv
Pipfile.lock: a list of packages that are installed in a virtual environment with hash code 
final_train.py: to train the final model
midterm_model: saved model using Pickle
midterm_project.ipynb: a storyline twinkling with data preprocessing,EDA,feature analysis,training models and evaluation
predict.py:Load  the model with Pickle and host with Flask,Requests
predict_test.py: send the request via Flask and testing the model

# How to run docker and pipenv
Install Docker for windows
copy scripts (final_train, predict and predict_test), pipenv file and Dockerfile to a folder

Install pipenv 

  
Install required packages
pipenv install numpy pandas scikit-learn flask waitress 


In terminal,create Docker image with the desired name.
[midterm-project]can be changed as u like.

docker build -t midterm-project .

Run Docker to load the model
docker run -it --rm -p 9696:9696 midterm-project
I used 9696:9696 because I host local 9696 port in Docker
In another terminal,run predict_test.py

  python predict_test.py

  
VOILA!! you can check the patient has heartdisease or not ^_^







