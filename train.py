
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
import os 
from imblearn.over_sampling import SMOTE
import pickle



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer


df = pd.read_csv('C:/Users/mulin/drug.csv')

df.columns = df.columns.str.lower()

cat_col = list(df.dtypes[df.dtypes=='object'].index)
num_col = list(df.dtypes[df.dtypes!='object'].index)

for c in cat_col:
     df[c] = df[c].str.lower().str.replace(' ', '_')

age_bin = [0,19,29,39,49,59,69,80]
age_category = ['<18','20','30','40','50','60','>60']
df['age_cal']= pd.cut(df['age'],bins=age_bin,labels=age_category)
df = df.drop(['age'],axis=1)

chemical_bin = [0,9,19,29,50]
chemical_category = ['<10','10-20','20-30','>30']
df['chemical_cal'] = pd.cut(df['na_to_k'],bins=chemical_bin,labels=chemical_category)
del df['na_to_k']

X = df.drop(["drug"], axis=1)
y = df["drug"]
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

X_train_re,y_train_re = SMOTE().fit_resample(X_train,y_train)
def train(df_train,y_train):
    train_dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_train_re,y_train_re = SMOTE().fit_resample(X_train,y_train)
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=15)
    rf.fit(X_train_re,y_train_re)
    return dv,rf

train(df_train,y_train)

def test(df_test,dv,model):
    test_dicts = df_test.to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

test(df_test,dv,rf)

output_file = f'model_capstone'
f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)