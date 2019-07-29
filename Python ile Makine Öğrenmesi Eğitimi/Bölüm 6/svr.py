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



# Veri Yukleme
veriler = pd.read_csv('maaslar.csv')

#DataFrame dilimleme(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,-1:]

#Linear Regression
#doğrusal mdel oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


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


#Görselleştirme
plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg.predict(x),color = "blue")
plt.show()

plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color = "blue")
plt.show()

plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(x)), color = "blue")
plt.show()

#tahminler 

print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))


from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color = "red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color = "blue")
plt.show()

print(svr_reg.predict(11))
print(svr_reg.predict(6.6))














    
    









