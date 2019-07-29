# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kod bölümü

#veri yükleme

veriler = pd.read_csv("veriler.csv")

#print(veriler)

boy = veriler[["boy"]]
#print(boy)

boykilo = veriler[["boy","kilo"]]
print(boykilo)

x = 10

class insan :
    boy = 180
    def kosmak(self,b) :
        return b+10

ali = insan()
print(ali.boy)
print(ali.kosmak(120))

liste = [1,2,3,4]

#veri ön işleme

















