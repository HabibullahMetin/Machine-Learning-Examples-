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
from sklearn.metrics import r2_score
import statsmodels.api as sm


# Veri Yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

#DataFrame dilimleme(slice)
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

print(veriler.corr())  # Corelation Matrixini çıkartır.

#Linear Regression
#doğrusal mdel oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("linear ols")
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())


print("Linear R2 değeri :")
print(r2_score(y,lin_reg.predict(x)))


#Polynomial Regression
#doğrusal olmayan  mdel oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(x)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


print("poly ols")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())
 
print("Polynomial R2 değeri :")
print(r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))


from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)


#Support Vector Machines
from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_olcekli,y_olcekli)



print("SVR ols")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print("Support Vector Machines R2 değeri :")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

#Decision Tree Regresyonu
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(x,y)



print("Decision Tree ols")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("Decision Tree R2 değeri :")
print(r2_score(y,r_dt.predict(x)))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
# n_estimators : Kaç tane decision tree çizileceğini belirtir.
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(x,y)



print("Random Forest ols")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


print("Random Forest R2 değeri :")
print(r2_score(y,rf_reg.predict(x)))

print("-----------------------------------------")

print("Linear R2 değeri :")
print(r2_score(y,lin_reg.predict(x)))

print("Polynomial R2 değeri :")
print(r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))

print("Support Vector Machines R2 değeri :")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("Decision Tree R2 değeri :")
print(r2_score(y,r_dt.predict(x)))

print("Random Forest R2 değeri :")
print(r2_score(y,rf_reg.predict(x)))   