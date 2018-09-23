import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

df = pd.read_csv("ACAR_INSURANCE.csv")

##Data Imputation
df['Communication'].fillna('other', inplace = True)
df['Job'].fillna(method = 'ffill', inplace = True)
df['Education'].fillna(method = 'ffill', inplace = True)
df['Outcome'].fillna('none', inplace = True)
#df['Outcome'].replace('other', 'failure', inplace = True)

##Features Engineering & Cleaning & Reselection
df['AvgBin'] = np.searchsorted([0, 60, 120, 180, 240, 300, 360], (df['DaysPassed'] / df['PrevAttempts'])).astype(np.int64)
df['LastContactMonth'] = [datetime.strptime(m, '%b').month for m in df.LastContactMonth]
df['WkDay'] = df.apply(lambda i:datetime.strptime('%s %s %s' %(2017, i.LastContactMonth, i.LastContactDay), '%Y %m %d'), axis = 1)
print(df.WkDay.dtypes)
print(df.WkDay)
df['WkDay'] = [datetime.isoweekday(a) for a in df.WkDay]
print(df.WkDay.dtypes)
print(df.WkDay)
df['BalancePositive'] = (df['Balance'] > 0).astype(object)
df['AgeBin'] = np.searchsorted([17, 30, 43, 57, 70, 83], df['Age']).astype(np.int64)
df = df.drop(columns=['Age', 'CallStart', 'CallEnd'])
df = df.loc[~(df['Balance'] > 70000)] ##Outliers Excluded
df = df.loc[~(df['PrevAttempts'] > 30)] ##Outliers Excluded
df = df.loc[~(df['Outcome'] == 'none')] ##No Outcome Excluded

##Label Encode & Concatenate
num = df.select_dtypes(include=['int64', 'float64'])
cat = (df.select_dtypes(include=['object'])).apply(LabelEncoder().fit_transform).astype(np.int64)
df = pd.concat([cat, num], axis = 1)

##Data Split - Target Variable Y: Outcome (Success or Failure) & Predictor Variables X: Remaining Features
y = df['Outcome']
print(y)
x=df.drop('Outcome', axis = 1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state=100, stratify = y)

##XGBoost Classifier, hyperparameter selected after RandomizedSearchCV
estimator = xgb.XGBClassifier()
parameter = {'n_estimators': [330],  'learning_rate': [0.01],  'max_depth': [5],  'min_child_weight': [7],
                      'subsample': [0.6],  'colsample_bytree': [0.7],  'gamma': [0.1],  'reg_alpha': [0.5]}

model = GridSearchCV(estimator, parameter, scoring = 'accuracy', cv = 10)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
print('=====Binary Classification Metric Evaluation=====')
print('Precision Score: %s' %(round((precision_score(ytest, ypred))*100, 1)), "%----------TP / (TP + FP)")
print('Recall Score: %s' %(round((recall_score(ytest, ypred))*100, 1)), "%---------------TP / (TP + FN)")
print('F Score (Harmonic Mean): %s' %(round((f1_score(ytest, ypred))*100, 1)),"%")

##Feature Importance Graph
dtrain = xgb.DMatrix(xtrain, ytrain)
clf = xgb.train(parameter, dtrain, 500)
xgb.plot_importance(clf, max_num_features = 25)
plt.show()
