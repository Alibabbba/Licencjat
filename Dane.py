# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:22:29 2021

@author: Mikołaj
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import periodogram, find_peaks

# =======================================
#   Różnica min/max oddechu
#   !Done! Periodogram/Transformata furiera scipy.signal.periodogram/numpy.fft 
#   !Done! Periodogram usrednic
#   !Done! Znalezc onset oddechu
#   Znalezc oddech zasypiajacych osob
#   
# =======================================


def Ready_panda(txt_file, path):
    df = pd.read_csv(path + txt_file, delimiter = "\t",header = None, names=["Time","Spasm","Breath",'Hbeat'])

    df.drop(df.head(2000).index, 
        inplace = True)
    df.drop(df.tail(2000).index, 
        inplace = True)   
    
    #peaks to indeksy wartosci lokalnych maximow (- dla odwrocenia grafu)
    peaks, _ = find_peaks(-df["Breath"].to_numpy(), distance = 1, prominence= 0.05)
    df["Onset_breath"] = 0
    for peak in peaks:
        # podmienia wartosci 0 na wartosci peaks
        df.iloc[peak, df.columns.get_loc("Onset_breath")] = df["Breath"].iloc[peak]

    df.dropna() 
    return df

def Data_info(database):
    print(database.dtypes)
    print(database.describe())
    print(database.head())
    print(database.tail())

def Show_sliced_graph(database ,start, how_much):
    how_much = start + how_much
    sliced_df = database[(database["Time"] < how_much) & (database["Time"] > start)]
    heart_beat = sliced_df[sliced_df["Hbeat"] != 0]
    onset_breath = sliced_df[sliced_df["Onset_breath"] != 0]

    fig, ax = plt.subplots(figsize=(4,4))
    sns.lineplot(x = "Time", y = "Breath", data = sliced_df, ax = ax)
    sns.scatterplot(x = "Time", y = "Breath", data = heart_beat, ax = ax, color= "r")
    sns.scatterplot(x = "Time", y = "Breath", data = onset_breath, ax = ax,
                    color= "g", s = 200, marker = "X")
        
    plt.show()
    plt.close()
    return fig
    
def Show_sliced_graph_movable(database):      
    loop = True
    while loop == True:
        for i in range(10):
            try:
                start = int(input("Start(int): "))
                break
            except:
                print("Invalid number")
                if i == 9:
                    start = 0
                continue

        for i in range(10):    
            try:            
                how_much = int(input("How long(int): ")) + start
                break
            except:
                print("Invalid number")
                if i == 9:
                    how_much = 10
                continue
        
        Show_sliced_graph(database, start, how_much)
        
        plt.close()
        
        val3 = input("Koniec? (y/n): ").lower().strip()
        if val3 == "y":
            loop = False

def Save_sliced_graph(database , start, how_much, name):
    how_much = start + how_much
    sliced_df = database[(database["Time"] < how_much) & (database["Time"] > start)]
    heart_beat = sliced_df[sliced_df["Hbeat"] != 0]

    fig, ax = plt.subplots(figsize=(4,4))
    sns.lineplot(x = "Time", y = "Breath", data = sliced_df, ax = ax)
    sns.scatterplot(x = "Time", y = "Breath", data = heart_beat, ax = ax, color= "r")
    sns.scatterplot(x = "Time", y = "Breath_event_min", data = sliced_df, ax = ax, color = 'y')
    os.chdir("C:/Users/Mikołaj/Desktop/Licencjat/Pics")
    plt.savefig(name + '.png')
    plt.close()
    
def Save_all_sliced_graphs(path, start, how_much):
    txt_file = os.listdir(path)
    for file in txt_file:
        my_database = Ready_panda(file, path)
        Save_sliced_graph(my_database, start, how_much, file)
    
def isNaN(num):
    return num != num

def Find_NaN(df):
    for index, row in df.iterrows():
        if isNaN(row["Hbeat"]):
            print(index)

def Frequency_mean(df_column,limit):
    asds = df_column.to_numpy()
    window_size = 10
    f, widmo =  periodogram(asds, fs = 1000)
    f = f[0:-window_size]
    widmo_mean = []
    for i in range(len(widmo) - window_size):
        widmo_mean.append(np.mean(widmo[i:i + window_size]))
    
    plt.plot(f,widmo_mean)
    plt.xlim(0, limit)
    plt.ylabel('Amplituda Widma Oddechu')
    plt.xlabel('Hz')
    return None
        

if __name__ == "__main__":
    
    path = "C:/Users/Mikołaj/Desktop/Licencjat/Dane/"
    txt_file = os.listdir(path)
    a = "Rob_1A_001Sel.txt"
    df = Ready_panda(txt_file[18], path)


    start = False

    if start == True:
        onset_timer = 0
        for index, row in df.iterrows():
            # IF event is not NaN
            if isNaN(row["Breath_event_min"]) == False :
                # onset_timer 
                if onset_timer < row["Time"]:
                    onset_timer = row["Time"] + 1.8
                    
                    df.loc[index + 2000, "Onset_breath"] = True                
                    
            print(index)
            if index == 15000:
                break
    
    



    

    # Show_sliced_graph(df ,10, 10)
    Show_sliced_graph_movable(df)
    # Data_info(df)    
    # Frequency_mean(df['Breath'],0.5)


