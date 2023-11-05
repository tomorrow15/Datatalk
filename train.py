import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle


raw_df = pd.read_csv('heart.csv')





raw_df.shape
raw_df.columns = raw_df.columns.str.lower()
raw_df.columns 
raw_df.isnull().sum()
raw_df.heartdisease.value_counts(normalize=True)
raw_df.heartdisease.mean()
raw_df.nunique()
raw_df.age.value_counts()


mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization
raw_df['oldpeak'] = mms.fit_transform(raw_df[['oldpeak']])
raw_df['age'] = ss.fit_transform(raw_df[['age']])
raw_df['restingbp'] = ss.fit_transform(raw_df[['restingbp']])
raw_df['cholesterol'] = ss.fit_transform(raw_df[['cholesterol']])
raw_df['maxhr'] = ss.fit_transform(raw_df[['maxhr']])
raw_df.head()
df_full_train, df_test = train_test_split(raw_df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.heartdisease .values
y_val = df_val.heartdisease.values
y_test = df_test.heartdisease.values
del df_train['heartdisease']
del df_val['heartdisease']
del df_test['heartdisease']

df_full_train.heartdisease.value_counts()






mutual_col = list(df_train.columns)
mutual_col




categorical_columns = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)
categorical_columns


# In[ ]:


numerical_columns = list(df_full_train.dtypes[df_full_train.dtypes != 'object'].index)
numerical_columns


le = LabelEncoder() 
df_full_train[categorical_columns] = df_full_train[categorical_columns].apply(lambda col: le.fit_transform(col)) 
df_full_train.head(5)





mutual_scores = []
for c in  df_full_train[mutual_col].columns: 
    score = round(mutual_info_score(df_full_train.heartdisease,df_full_train[c]),3)
    mutual_scores.append(score)
    print(f"mutual score for {c} is {score}")





import matplotlib.pyplot as plt

sorted_mutual_scores, sorted_mutual_col_names = zip(*sorted(zip(mutual_scores, mutual_col)))
plt.bar(sorted_mutual_col_names, sorted_mutual_scores)
plt.xlabel("Features")
plt.ylabel("Mutual Information Scores")
plt.title("Mutual Information Scores for Features")
plt.xticks(rotation='vertical')
plt.show()




for column in categorical_columns:
    unique_categories = df_full_train[column].unique() # this is important
    print(f"Heart disease rate for {column}:")
    for category in unique_categories:
        mean = df_full_train[df_full_train[column] == category].heartdisease.mean()
        print(f"{category}: {mean}")
    print("\n")




global_heartdisease = df_full_train.heartdisease.mean()


for c in categorical_columns:
    print(c)
    df_group = df_full_train.groupby(c).heartdisease.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] -global_heartdisease
    df_group['risk'] = df_group['mean'] /global_heartdisease
    df_group
    print(df_group)
    print()


numeric_columns = list(df_full_train.dtypes[df_full_train.dtypes != 'object'].index)
numeric_columns




data_numeric = df_full_train[numeric_columns]
data_numeric.describe()





data_numeric.corr()





plt.figure(figsize=(9, 6))
sns.heatmap(data_numeric.corr())
plt.title('Heatmap showing correlations between numerical data')
plt.show();




threshold = 0.1


columns_to_drop = [col for col in df_full_train.columns if mutual_info_score(df_full_train.heartdisease,df_full_train[col])< threshold]


df_full_train = df_full_train.drop(columns=columns_to_drop)
df_full_train




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




dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)




model = LogisticRegression(solver='lbfgs')
model.fit(X_train,y_train)




y_pred = model.predict_proba(X_val)[:, 1]




heartdisease_decision = (y_pred >= 0.5)





(y_val == heartdisease_decision).mean()





df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = heartdisease_decision.astype(int)
df_pred['actual'] = y_val
df_pred





roc_auc_score(y_val, y_pred)



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




scores = []
depths = [4,5,6]

for depth in depths: 
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        x_dict = df_train.to_dict(orient='records')
        dv= DictVectorizer(sparse=False)
        x_train = dv.fit_transform(x_dict)
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(x_train, y_train)
        val_dicts = df_val.to_dict(orient='records')
        x_val = dv.transform(val_dicts)
        y_pred = dt.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((depth, s, auc))




columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)





df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)





sns.heatmap(df_scores_pivot, fmt=".3f");





dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
dt.fit(X_train, y_train)




val_dicts = df_val.to_dict(orient='records')
x_val = dv.transform(val_dicts)
y_pred = dt.predict_proba(x_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc





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




scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))





columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)





df_scores_pivot = df_scores.pivot(index='n_estimators', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)





sns.heatmap(df_scores_pivot, fmt=".3f");




for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend();




max_depth = 15





scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))





columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)





sns.heatmap(df_scores_pivot, fmt=".3f");




colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend();





min_samples_leaf = 3




scores = []
for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))





columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)





rf = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,random_state=1)

rf.fit(x_train, y_train)




y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc





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





x_dict = df_train.to_dict(orient='records')
dv= DictVectorizer(sparse=False)
x_train = dv.fit_transform(x_dict)
val_dicts = df_val.to_dict(orient='records')
x_val = dv.transform(val_dicts)




features = dv.get_feature_names_out()
features





features = dv.get_feature_names_out()
features = features.tolist()
dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(x_val, label=y_val, feature_names=features)





xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)





y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)





watchlist = [(dtrain, 'train'), (dval, 'val')]





get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=10,\n                  verbose_eval=5,\n                  evals=watchlist)\ny_pred = model.predict(dval)\n")




roc_auc_score(y_val, y_pred)





s = output.stdout





print(s[:200])




def parse(output):
    scores = []
    for line in output.stdout.strip().split('\n'):
            a,b,c = line.split('\t')
            itr = int(a.strip('[]'))
            train = float(b.split(':')[1])
            val = float(c.split(':')[1])
            
            scores.append((itr,train,val))
    columns = ['iteration','train_auc','val_auc']
    df_scores = pd.DataFrame(scores,columns=columns)
    return df_scores





df_score = parse(output)





plt.plot(df_score.iteration, df_score.train_auc, label='train')
plt.plot(df_score.iteration, df_score.val_auc, label='val')
plt.legend();





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




dv, model,y_pred




output_file =  'mid_term_model'





with open(output_file,'wb')as f_out:
    pickle.dump((dv,model),f_out)
print(f'the model is saved to {output_file}')






