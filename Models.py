

from typing import List
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV  


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline



from pactools import simulate_pac
from pactools.grid_search import ExtractDriver, AddDriverDelay
from pactools.grid_search import DARSklearn, MultipleArray
from pactools.grid_search import GridSearchCVProgressBar


#Import data 
data=pd.read_csv('data_complete.csv')

data=pd.read_csv('/Users/adamrudolfsson/Python_folder/data_complete.csv')

data.drop('Unnamed: 0.1', inplace=True, axis=1)
data.drop('Unnamed: 0', inplace=True, axis=1)
data.shape
data.head()



#Classification porbelm 
#####################################################
#Splitting the data

X = data[['positive','negative','neutral']]
y= pd.pandas.DataFrame(data['TF_AR'])
y=np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)



#####  SVM 
from sklearn.svm import SVC  

clf=SVC()
grid_search = {'C': [0.01],
              'gamma': [100],
              'degree':[8],
              'kernel': ['poly']} 

model = GridSearchCVProgressBar(estimator=clf, param_grid=grid_search, cv=10, 
                              refit=True, verbose=10, n_jobs = -1) 
model.fit(X_train,y_train)

predictionforest1 = model.best_estimator_.predict(X_test)
cm1=confusion_matrix(y_test,predictionforest1)
cr1=classification_report(y_test,predictionforest1)
acc1 = accuracy_score(y_test,predictionforest1)

print(cr1)



####  Random Forest 
from sklearn.ensemble import RandomForestClassifier

grid_search= {'criterion': ['entropy', 'gini'],
               'max_depth':  [3,4,5,6,7],
               'max_features':[10],
               'min_samples_leaf': [1,2,3,4,5,6,7],
               'min_samples_split': [1],
               'n_estimators': [150]}

clf = RandomForestClassifier()
model = GridSearchCVProgressBar(estimator = clf, param_grid = grid_search, 
                               cv = 10, verbose= 10, n_jobs = -1)
model.fit(X_train,y_train)

predictionforest = model.best_estimator_.predict(X_test)
cm2=confusion_matrix(y_test,predictionforest)
cr2=classification_report(y_test,predictionforest)
acc2 = accuracy_score(y_test,predictionforest)

print(cr2)





###   Neural network 
#pip3 install pactools
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Function to create model, required for KerasClassifier
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=3, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
grid_search= {
    'batch_size' : [20],
    'epochs' : [5],
    'neurons' : [2,3,4,5]}

clf=KerasClassifier(build_fn=create_model)
model= GridSearchCVProgressBar(estimator=clf, param_grid=grid_search, n_jobs=-1,verbose= 10, cv=10,refit=True)
model.fit(X_train,y_train)

predictionforest = model.best_estimator_.predict(X_test)
cm3=confusion_matrix(y_test,predictionforest)
cr3=classification_report(y_test,predictionforest)
acc3 = accuracy_score(y_test,predictionforest)


print(cr3)


####  K-Nerest nabour
from sklearn.neighbors import KNeighborsClassifier

grid_search= {
    'n_neighbors': [3,4,5,6],
    'p': [1,2],
    'weights': ['uniform','distance'],
    'metric': ['minkowski', 'chebyshev']}

clf = KNeighborsClassifier()

model= GridSearchCVProgressBar(estimator=clf, param_grid=grid_search,  n_jobs = -1, cv = 10,refit=True, verbose=10)

model.fit(X_train,y_train)
predictionforest4 = model.best_estimator_.predict(X_test)
cm4=confusion_matrix(y_test,predictionforest4)
cr4=classification_report(y_test,predictionforest4)
acc4 = accuracy_score(y_test,predictionforest4)
print(cr4)






####   XG Boost
X = data[['positive','negative','neutral']]
y= pd.pandas.DataFrame(data['TF_AR'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

import xgboost as xgb
from sklearn.datasets import load_boston
 
grid_search= {
    'objective': ['binary:logistic'],
    'gamma':[1.1],
    'max_depth': [3,4,5], 
    'colsample_bylevel': [0.45,0.5,0.55],
    'learning_rate': [0.06,0.05],
    'n_estimators':[150]}
 
clf = xgb.XGBClassifier()
model= GridSearchCVProgressBar( estimator=clf, param_grid=grid_search,  n_jobs = -1, cv = 10, refit=True, verbose=10)
model.fit(X_train,y_train)

predictionforest5 = model.best_estimator_.predict(X_test)
cm5=confusion_matrix(y_test,predictionforest5)
cr5=classification_report(y_test,predictionforest5)
acc5 = accuracy_score(y_test,predictionforest5)
print(cm5)
print(cr5)
































































###Linear regresion problem 
#########################################################################


X = data[['positive','negative','neutral']]
y= pd.pandas.DataFrame(data['AR'])
y['AR']  = y["AR"].str.replace(",",'.')
y=np.ravel(y,)
y = [ast.literal_eval(i) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)




##### Random Forest
from ast import literal_eval
from sklearn import metrics
import ast
from sklearn.ensemble import RandomForestRegressor

grid_search= {'bootstrap': ['True', 'False'],
               'max_depth':  list(np.linspace(1, 20, 10, dtype = int)),
               'max_features':list(np.linspace(1, 20, 10, dtype = int)),
               'min_samples_leaf': list(np.linspace(2, 20, 10, dtype = int)),
               'min_samples_split': list(np.linspace(2, 20, 10, dtype = int)),
               'n_estimators': list(np.linspace(150, 200, 1, dtype = int))}


rf = RandomForestRegressor()
model = GridSearchCVProgressBar(estimator = rf, param_grid = grid_search, 
                               cv = 10, verbose= 0, n_jobs = -1)
model.fit(X_train,y_train)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

best_random = model.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

y_pred = best_random.predict(X_test)
print('MSE: ', mean_squared_error(y_test, y_pred)) 



























