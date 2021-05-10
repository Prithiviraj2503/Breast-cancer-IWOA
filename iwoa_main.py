"""
Created on Fri Feb 19 21:58:37 2021
Project Title: Feature Selection Using Improved Whale Optimization Algorithm for High Dimensional Microarray Data

@Author: Prithiviraj K
Reg. No: 810017205062
Final year-IT-'B'.

@Guided By: Dr. S. Sathiya Devi
Department of Information Technology
University College of Engineering, BIT campus- Tiruchirappalli.
"""
#import necessary libraries
import pandas as pd
import numpy as np
from iwoa import woa 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


# load data
data  = pd.read_csv('breastCancer.csv', low_memory=False, index_col=0).T

#converting categorical class into numerical
data.y[data.y == 'luminal'] = 1 #set all the luminal type as 1
data.y[data.y == 'non-luminal'] = 0 #set all the non-luminal type as 0

#Splitting the entire data into features abd targets
features=np.asarray(data.drop(columns='y'))
label=np.asarray(data['y'])
target=label.astype('int')

# split data into train & test(30%)
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
N = 50  #Number of Whales
maxIter  =  30  #Number of Iteration
opts = {'fold':fold, 'N':N, 'T':maxIter} #make it dictionary for easy access

# perform feature selection
fselect = woa(features, target, opts)
sel_feat= fselect['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sel_feat]
y_train   = ytrain.reshape(num_train)
x_valid   = xtest[:, sel_feat]
y_valid   = ytest.reshape(num_valid) 

#Fit our data into the SVM model
#gb= GradientBoostingClassifier(learning_rate=0.1, max_depth=4, n_estimators=100)
model = SVC(kernel="rbf")
model.fit(x_train, y_train)
#Prediction 
y_pred    = model.predict(x_valid)

# plot convergence
print("\nPLOTTING CONVERGENCE CRITERIA")
curve   = fselect['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness (Error rate)')
ax.set_title('Improved Whale Optimization Algorithm')
ax.grid()
plt.show()

#Performance evaluation
print("\nPERFORMANCE EVALUATION")
print("\nConfusion Matrics", metrics.confusion_matrix(y_valid, y_pred))
Accuracy=metrics.accuracy_score(y_valid, y_pred)
print("Accuracy:", 100 * Accuracy)
print("\nError rate:",1-Accuracy)
print("\nPrecision", metrics.precision_score(y_valid, y_pred))
print("\nRecall", metrics.recall_score(y_valid, y_pred))
print("\nROC curve")

fpr, fpr, _ = metrics.roc_curve(y_valid, y_pred)
plt.plot(fpr, fpr, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print("\nAUC-ROC score", metrics.roc_auc_score(y_valid, y_pred))