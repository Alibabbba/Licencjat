# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:37:51 2021

@author: Mikołaj
"""

txt_1A = """
          1A       1.000  0.009  0.013  0.114  0.892"""
txt_1B = """
          1B       0.114  0.975  0.890  1.000  0.248"""
txt_2A = """
          2A       0.013  0.846  1.000  0.890  0.169"""
txt_2B = """
          2B       0.009  1.000  0.846  0.975  0.142"""
txt_2C = """ 
          2C       0.892  0.142  0.169  0.248  1.000"""
txt_avg = """
weighted avg       0.61      0.57      0.57        14"""

txt_1B = txt_1B.replace("2A", "1B")
txt_2A = txt_2A.replace("1B", "2A")
txt_avg = txt_avg.replace("weighted avg", "$M$")


txt_list = [txt_1A, txt_1B, txt_2A, txt_2B, txt_2C, txt_avg]

print("\hline")
for i in txt_list:
    print(" & ".join(i.split()), "\\\\")
    print("\hline")
