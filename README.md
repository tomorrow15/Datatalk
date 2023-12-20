# Capstone project 1  ML Zoomcamp 2023: Drug Classification 
# Introduction 

This dataset contains information about drug classification based on patient general information. Machine learning model is needed in order to predict the outcome of the drugs.

In this project, I used multiple machine learning models to predict drug classification, including logistic regression, decision tree, random forest. I selected the model with the best accuracy score and deployed it in a Docker container.

# Dataset

This is drug classification project that uses drug classification dataset from Kaggle datasets.  Here is the link to it:https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data

# Feature Information

Age: age of the patient 

Sex: sex of the patient 

BP : blood pressure level of the patient

Cholesterol : Cholesterol Levels of the patient

Na to Potassium Ration : Sodium to potassium ratio in blood

Drug : Drug type

# File Description

Dockerfile : Drug Classification service  including python image - python:3.11-slim,pipenv and hosting the server with waitress 

Pipfile: it describe which packages are installed in pipenv

Pipfile.lock: a list of packages that are installed in a virtual environment with hash code 

train.py: to train the final model

model_capstone: saved model using Pickle

drug_classification.ipynb: a storyline twinkling with data preprocessing,EDA,feature analysis,training models and evaluation

predict.py:Load  the model with Pickle and host with Flask,Requests

test.py: send the request via Flask and testing the model

# How to run docker and pipenv
Install Docker for windows

copy scripts (final_train, predict and predict_test), pipenv file and Dockerfile to a folder

>Install pipenv

  
Install required packages

>pipenv install numpy pandas scikit-learn flask waitress 


In terminal,create Docker image with the desired name.

[model_capstone]can be changed as u like.

>docker build -t capstone .

Run Docker to load the model

>docker run -it --rm -p 9696:9696 capstone

I used 9696:9696 because I host local 9696 port in Docker

In another terminal,run predict_test.py

>python predict.py

  
Now!you can predict the drugs type that might be suitable for the patient ^_^







