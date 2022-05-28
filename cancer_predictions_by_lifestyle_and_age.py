# -*- coding: utf-8 -*-
"""
Created on Thu May 26 05:30:14 2021

@author: KReuZ_o13
"""
#Hello, there stranger. Today, we're analysing some cancer data
#First things first, let's get our packages
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#With that sorted, time to import our data!
cancer_data = pd.read_csv("C:/Users/ADMIN1/Desktop/Projects/Python Projects/Machine_Learning/cancer patient data sets.csv")

#Let's try to get a sense of what we have on hand; first step - seeing the column heads and first few instances of data
cancer_data.head()

#Okay, let's see the column names, the number of instances and the datatypes we're working with!
cancer_data.columns
cancer_data.shape
cancer_data.info

#Now that we know that we have 25 columns with 1000 people and no nil values, let's see the data analysed via statistics!
cancer_data.describe().T

#Interesting  to note the means and deviations, right?
#Now we seek to see the data.
#First, the quantities of people at varying risks via the all mighty pie chart!
#We need to find out the factored data at risk and their numbers, so...
sbn.set(rc = {'figure.figsize': (10,7)})
labels = list(cancer_data.Level.unique())
sizes = list(cancer_data.Level.value_counts())

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct = '%1.2f%%', startangle = 65)
plt.title('Patient distribution according to cancer risk')

#Looks like the data we have has an almost equal disrtibution between the three! Nice!
#Let's get a heatmap in to see more!
#Remember to distinguish your plots before they fuse
fig2 = plt.figure(figsize = (13,8))
sbn.heatmap(cancer_data.corr(),cmap='coolwarm',annot=True);
plt.title('Heatmap showing the high risk combinations')

#We can already see some bad combinations, such as occupational hazard and genetic risk
#Now let's see how the data looks like if we look at age and gender!
fig3, ax3 = plt.subplots()
plot = sbn.countplot(data = cancer_data, x='Level', hue='Gender', palette=['blue','green'])
plt.title('Patient distribution according to cancer risk and gender')

#Seems that men are at a higher risk from the data.
#But how many men are there compared to women?
#Time for another pie chart!
fig4, ax4 = plt.subplots()
sbn.set(rc = {'figure.figsize': (10,7)})
labels = list(cancer_data.Gender.unique())
sizes = list(cancer_data.Gender.value_counts())
ax4.pie(sizes, labels=labels, autopct = '%1.2f%%', startangle = 65)
plt.title('Patient distribution according to gender')

#Relatively balanced as well, nice! Probably should have done this first
#To make life easier, I'll convert the level data into an int object, and drop the patient ID since we're not really using it
cancer_data["Level"].replace(["Low", "Medium", "High"], ["0", "1", "2"], inplace=True)
cancer_data["Level"] = cancer_data["Level"].astype(int)
cancer_data.drop(["Patient Id"], axis = 1, inplace= True)
cancer_data.head

#I think I'd like to see the probability density so....
fig5, ax5 = plt.subplots()
cancer_data.plot.kde(figsize = (20,8));
plt.title('Probability density')

#Time for some ML!
#First, linear regression as is proper :)
#Step one: process your data! We split the data into training and test sets, then we normalize it to remove any outliers
X = cancer_data.drop(["Level"], axis = 1)
y = cancer_data["Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Then we train (after fitting)!
lr = LogisticRegression()
lr.fit(X_train, y_train) 
y_lr = lr.predict(X_test)

#Now, accurate are we?
score_LR = accuracy_score(y_test, y_lr)
print(score_LR)
print(classification_report(y_test, y_lr))

#Now to see how accurate the model really was!
resultat_1 = confusion_matrix(y_test, y_lr)
fig6, ax6 = plt.subplots()
sbn.heatmap(resultat_1, annot=True)
plt.title('Confusion Matrix for Linear Regression')
plt.xlabel('y_test')
plt.ylabel('y_lr') 

#Now we redo this with other models to find the one true ring.
#First, SVM
A = cancer_data.drop(["Level"], axis = 1)
b = cancer_data["Level"]

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2)

sv = svm.SVC()
sv.fit(A_train, b_train)
sv.score(A_test, b_test)
b_svm = sv.predict(A_test)

#Now, accurate are we?
score_SVM = accuracy_score(b_test, b_svm)
print(score_SVM)
print(classification_report(b_test, b_svm))

#Now to see how accurate the model really was!
resultat_2 = confusion_matrix(b_test, b_svm)
fig7, ax7 = plt.subplots()
sbn.heatmap(resultat_2, annot=True)
plt.title('Confusion Matrix for SVM')
plt.xlabel('b_test')
plt.ylabel('b_svm') 

#Next, KNNs
C = cancer_data.drop(["Level"], axis = 1)
d = cancer_data["Level"]

C_train, C_test, d_train, d_test = train_test_split(C, d, test_size = 0.2)

knn = KNeighborsClassifier()
knn.fit(C_train, d_train)
knn.score(C_test, d_test)
d_knn = knn.predict(C_test)

#Now, accurate are we?
score_KNN = accuracy_score(d_test, d_knn)
print(score_KNN)
print(classification_report(d_test, d_knn))

#Now to see how accurate the model really was!
resultat_3 = confusion_matrix(d_test, d_knn)
fig8, ax8 = plt.subplots()
sbn.heatmap(resultat_3, annot=True)
plt.title('Confusion Matrix for KNN')
plt.xlabel('d_test')
plt.ylabel('d_knn')

#It's interesting to note that the KNN and regression gives a higher accuracy score than the SVM
