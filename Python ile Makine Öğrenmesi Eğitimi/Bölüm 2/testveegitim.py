# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:20:30 2019

@author: hp
"""

#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kod bölümü

#veri yükleme

veriler = pd.read_csv("eksikveriler.csv")

#print(veriler)

boy = veriler[["boy"]]
#print(boy)

boykilo = veriler[["boy","kilo"]]
#print(boykilo)

x = 10

class insan :
    boy = 180
    def kosmak(self,b) :
        return b+10

ali = insan()
#print(ali.boy)
#print(ali.kosmak(120))

liste = [1,2,3,4]

#eksik veriler

"""
eksik verileri tamamlamak için o kolonun ortalamasını eksik verilere yazarız.
"""
#sci - kit learn
#impute = töhmet 

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "NaN",strategy = "mean",axis = 0) 
#Sayısal veriler için geçerlidir.
# Sayısal verileri diğer verilerden ayırmak gerekir.
#iloc = verileri çeken fonksiyon
Yas = veriler.iloc[:,1:4].values # 1 kolondan başla 4 e kadar(4 dahil değil)
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)


ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

#kolonlama
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = "all")
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


sonuc = pd.DataFrame(data = ulke, index = range(22),columns = ["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22),columns = ["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])
print(sonuc3)

#concat : verileri(dataframeleri) birleştirir.

s = pd.concat([sonuc,sonuc2],axis = 1) # axis = 1 ile satır bazlı birleştirme yapar
print(s)

s2 = pd.concat([s,sonuc3],axis = 1)
print(s2)

from sklearn.cross_validation import train_test_split  # farklı yöntemlerde var

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state = 0)
"""
s = train in alınacağı yerler
sonuc3 = bağımsız değişkenlerin alınacağı yerler
random_state : veriyi nasıl böleceği (random böler 0 yaparsak)

"""



























