# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:02:46 2019

@author: hp
"""

#1.kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("satislar.csv")
print(veriler)

aylar = veriler[["Aylar"]]
print(aylar)
satislar = veriler[["Satislar"]]
print(satislar)
# ilk ":"'dan önceki kısım hiçbir şey girilmezse tüm satırları alır
"""satislar2 = veriler.iloc[:,:1].values 
print(satislar2) """



#Verilerin Eğitim ve Test için bölünmesi
from sklearn.cross_validation import train_test_split  # farklı yöntemlerde var

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size = 0.33,random_state = 0)

"""
#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler #Standarlaştırma(Standardization)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

#model inşası (Linear Regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#fit : modeli inşa eder
lr.fit(x_train,y_train)  #X_train'den Y_train'i öğrendi

tahmin = lr.predict(x_test)    #X_test'den de tahmin sonuçlarını çıkardı

x_train = x_train.sort_index()
y_train = y_train.sort_index()  #indexleri sıralıyoruz verileri değil.

plt.plot(x_train,y_train)   
plt.plot(x_test,tahmin)    

plt.title("aylara göre satış")  
plt.xlabel("aylar")
plt.ylabel("satışlar")                  






















