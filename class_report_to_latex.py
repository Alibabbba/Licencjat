# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:37:51 2021

@author: Mikołaj
"""

txt_1A = """
          1A       1.00      0.20      0.33         5"""
txt_1B = """
          2B       0.50      0.60      0.55         5"""
txt_2A = """
          2A       0.43      0.75      0.55         4"""
txt_2B = """
    2B     0.66      0.50      0.47        14"""
txt_2C = """ 
      2C  0.80      0.80      0.80         5"""
txt_avg = """
weighted avg       0.66      0.50      0.47        14"""

txt_1B = txt_1B.replace("2A", "1B")
txt_2A = txt_2A.replace("1B", "2A")
txt_avg = txt_avg.replace("weighted avg", "$M$")


txt_list = [txt_1A, txt_1B, txt_2A, txt_2B, txt_2C, txt_avg]

print("\hline")
for i in txt_list:
    print(" & ".join(i.split()), "\\\\")
    print("\hline")
