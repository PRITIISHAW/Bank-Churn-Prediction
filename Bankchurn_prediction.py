# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:39:19 2019

@author: Priti
"""

import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns


# load the data
df = pandas.read_csv(r"C:\Users\Priti\Desktop\Bank_churn_modelling.csv")
df.shape
# drop some columns which are not so important based on domain knowledge

df.drop(['CustomerId',"Surname","RowNumber"],axis=1,inplace=True)
# axis =1 to drop a cloumn and axis = 0 to drop a row
# inplace = True => permanent changes to the dataframe
df.shape
##############################################################
# check for missing values - wether data has missing values or not
df.isnull().sum()
# our data doesnt have missing values so we are good to go ahead

# check for duplicates
df.duplicated().sum()
# df.drop_duplicates(inplace=True)
# we dont dont have duplicate customer entries so we are good to
# go ahead

###########################################################
# Analytics using Data Visualization
# impact of creditscore v/s exited
# probability distribution analysis
plt.figure(figsize=(12,5))
sns.distplot(df["CreditScore"][df["Exited"]==0])
sns.distplot(df["CreditScore"][df["Exited"]==1])
plt.legend(['0','1'])
plt.show()

# age v/s exited
plt.figure(figsize=(12,5))
sns.distplot(df["Age"][df["Exited"]==0])
sns.distplot(df["Age"][df["Exited"]==1])
plt.legend(['0','1'])
plt.show()

#################################################
#################################################
# georgaphy v/s exited
plt.figure(figsize=(6,4))
sns.countplot(df['Geography'])
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(df['Geography'][df['Exited']==1])
plt.show()


# gender v/s exited
plt.figure(figsize=(6,4))
sns.countplot(df['Gender'])
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(df['Gender'][df['Exited']==1])
plt.show()


# HasCrCard v/s exited
plt.figure(figsize=(6,4))
sns.countplot(df['HasCrCard'])
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(df['HasCrCard'][df['Exited']==1])
plt.show()

#correlation analysis
cor = df.corr()
# visualize using heatmap
plt.figure(figsize=(12,10))
sns.heatmap(cor,annot=True,cmap="coolwarm")
plt.show()

###########################################
############################################
# take important features based on correlation analysis

df = df[["Age","Geography","Gender","Balance",
        "IsActiveMember","Exited"]]

# separating features and the label
x = df.drop(["Exited"],axis=1)
y = df["Exited"]

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
x["Geography"] = le1.fit_transform(x["Geography"])

le2 = LabelEncoder()
x["Gender"] = le2.fit_transform(x["Gender"])

# onehotencoding for geography column
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()

# scaling of featuers
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

################################################
#############################################
# train test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

# apply Machine Learning - logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# train the model using train data
model.fit(xtrain,ytrain)

'''
Age = 45
geography = France
gender = Female
balance = 456212
Isactivemember = 1
'''
ip = [[1,0,0,45,0,456212,1]]
model.predict(ip)
# check accuracy of model on the test data
# get the prediction from model for 2000 customers
ypred = model.predict(xtest)
# check accuracy by comparing ypred- prediction with ytest-actual
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)

# what is recall here - how much business purpose achieved
from sklearn.metrics import recall_score
recall_score(ytest,ypred)


###############################################
from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
#train the model
model.fit(xtrain,ytrain)
# check accuracy of model on the test data
# get the prediction from model for 2000 customers
ypred = model.predict(xtest)
# check accuracy by comparing ypred- prediction with ytest-actual
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)

# what is recall here - how much business purpose achieved
from sklearn.metrics import recall_score
recall_score(ytest,ypred)

######################################################
######################################################

# export the model
from sklearn.externals import joblib
joblib.dump(model,"chrun_model.pkl")








