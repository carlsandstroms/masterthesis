from tkinter import Y
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
%matplotlib inline


matrix=pd.read_csv('data_TTN.csv')
matrix.drop('Unnamed: 0', inplace=True, axis=1)
matrix.shape
matrix.head()

#X = matrix[['year','quarter','positive','negative','neutral']]

#X = matrix[['year','quarter']]










###Linear reg 
#pip3 install seaborn
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

X = matrix[['positive','negative','neutral']]
X["positive"] = X["positive"].str.replace("[\]\[]",'')
X["negative"] = X["negative"].str.replace("[\]\[]",'')
X["neutral"]  = X["neutral"].str.replace("[\]\[]",'')
y= pd.pandas.DataFrame(matrix['AR'])
y['AR']  = y["AR"].str.replace(",",'.')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
X_train=X
X_test=X
y_train=y
y_test=y


mlr = LinearRegression()  
mlr.fit(X_train, y_train)
list(zip(X, mlr.coef_))

y_pred_mlr= mlr.predict(X_test)
print("Prediction for test set: {}".format(y_pred_mlr))

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})



meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(X,Y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)





#####SVM 

X = matrix[['positive','negative','neutral']]
X["positive"] = X["positive"].str.replace("[\]\[]",'')
X["negative"] = X["negative"].str.replace("[\]\[]",'')
X["neutral"]  = X["neutral"].str.replace("[\]\[]",'')
y=matrix["TF_AR"]


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))




###Dec tree
X = matrix[['positive','negative','neutral']]
X["positive"] = X["positive"].str.replace("[\]\[]",'')
X["negative"] = X["negative"].str.replace("[\]\[]",'')
X["neutral"]  = X["neutral"].str.replace("[\]\[]",'')
y=matrix["AR"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
X_train=X
X_test=X
y_train=y
y_test=y

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.tree import plot_tree
plt.figure(figsize=(10,8), dpi=150)
plot_tree(model, feature_names=X.columns)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))










## CLASS TREE

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X = matrix[['positive','negative','neutral']]
#X = matrix[['year','quarter','positive','negative','neutral']]

X["positive"] = X["positive"].str.replace("[\]\[]",'')
X["negative"] = X["negative"].str.replace("[\]\[]",'')
X["neutral"]  = X["neutral"].str.replace("[\]\[]",'')
y=matrix["TF_TR"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
X_train=X
X_test=X
y_train=y
y_test=y


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))







###Random forets 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



X = matrix[['positive','negative','neutral']]

X["positive"] = X["positive"].str.replace("[\]\[]",'')
X["negative"] = X["negative"].str.replace("[\]\[]",'')
X["neutral"]  = X["neutral"].str.replace("[\]\[]",'')
y=matrix["TF_AR"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
X_train=X
X_test=X
y_train=y
y_test=y

clf = RandomForestClassifier(max_depth=8, random_state=0)
clf= clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))