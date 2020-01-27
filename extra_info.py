# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:45:46 2019

@author: Priti
"""

from sklearn.externals import joblib
model = joblib.load("chrun_model.pkl")

'''
Age = 65
geography = France
gender = Female
balance = 456212
Isactivemember = 0
'''
ip = [[1,0,0,65,0,456212,0]]
model.predict(ip)