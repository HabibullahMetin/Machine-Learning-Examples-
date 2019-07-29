#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #bağımsız değişken
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)
#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Buradan itibaren sınıflandırma algoritmaları başlar
#1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(x_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)
#karmaşıklık matrisi
cm = confusion_matrix(y_test,y_pred)
print("Logistic Regression")
print(cm)
# 2.K - Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1, metric = "minkowski")

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("K-NN")
print(cm)
# 3. Support Vector Machines with Classification
from sklearn.svm import SVC
svc = SVC(kernel = "poly")

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)
# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("GNB")
print(cm)
# 5. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy")

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)
# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators =10, criterion="entropy")
#n_estimators'a göre başarı oranı değişir.
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("RFC")
print(cm)

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

#ROC fpr tpr thold değerleri
from sklearn import metrics
fpr ,tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label="e")
print(fpr)
print(tpr)




    









    

