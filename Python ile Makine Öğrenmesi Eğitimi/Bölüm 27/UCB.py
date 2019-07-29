# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:35:29 2019

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

import random 

#Random Selection
"""
N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N) :
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n.satır = 1 ise odul 1
    toplam = toplam + odul


plt.hist(secilenler)
plt.show()
"""
import math
#UCB

N = 10000 #10.000 işlem(tıklama)
d = 10 #toplam 10 ilan var
oduller = [0]*d #başlangıçta tüm ilanların ödülleri 0 R değeri
tiklamalar = [0]*d # N değeri o ana kadarki tıklamalar
toplam_odul  = 0
secilenler = []

for n in range(0,N) :
    ad = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0,d) :
        if(tiklamalar[i] > 0) :
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else :
            ucb = N*10
        if max_ucb < ucb : #maxtan büyük bir ucb  çıktı
            max_ucb = ucb
            ad  = i
            
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad] # verilerdeki n.satır = 1 ise odul 1
    oduller[ad] = oduller[ad] + odul
    toplam_odul = toplam_odul + odul


print("Toplam odul :",toplam_odul)
plt.hist(secilenler)
plt.show()








