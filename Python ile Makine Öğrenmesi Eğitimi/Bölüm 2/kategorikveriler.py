# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:36:14 2019

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

























