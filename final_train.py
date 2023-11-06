import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  LabelEncoder
from sklearn.metrics import roc_auc_score


raw_df = pd.read_csv('heart.csv')
raw_df.columns = raw_df.columns.str.lower()



mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

raw_df['oldpeak'] = mms.fit_transform(raw_df[['oldpeak']])
raw_df['age'] = ss.fit_transform(raw_df[['age']])
raw_df['restingbp'] = ss.fit_transform(raw_df[['restingbp']])
raw_df['cholesterol'] = ss.fit_transform(raw_df[['cholesterol']])
raw_df['maxhr'] = ss.fit_transform(raw_df[['maxhr']])





df_full_train, df_test = train_test_split(raw_df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.heartdisease .values
y_val = df_val.heartdisease.values
y_test = df_test.heartdisease.values

del df_train['heartdisease']
del df_val['heartdisease']
del df_test['heartdisease']

categorical_columns = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)
numerical_columns = list(df_full_train.dtypes[df_full_train.dtypes != 'object'].index)

le = LabelEncoder()
df_full_train[categorical_columns] = df_full_train[categorical_columns].apply(lambda col: le.fit_transform(col)) 

dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='lbfgs')
model.fit(X_train,y_train)

y_pred = model.predict_proba(X_val)[:, 1]
heartdisease_decision = (y_pred >= 0.5)
def final_train(df_train, y_train, C=1.0):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(dicts)

    model =LogisticRegression(solver='lbfgs')
    model.fit(x_train, y_train)
    
    return dv, model
def final_predict(df_test, dv, model):
    dicts = df_test.to_dict(orient='records')

    x_test = dv.transform(dicts)
    y_pred = model.predict_proba(x_test)[:, 1]

    return y_pred

dv, model = final_train(df_full_train, df_full_train.heartdisease.values, C=1.0)
y_pred = final_predict(df_test, dv, model)

output_file =  'midterm_model'

with open(output_file,'wb')as f_out:
   pickle.dump((dv,model),f_out)
print(f'the model is saved to {output_file}')







