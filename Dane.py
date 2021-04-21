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
#   Podzielic hbeat na klasy
#   opisać jak rozóżniałem ręcznie i dlaczego lub ucinać
#   Synchrogramy i korelogramy te drugie
#   Zrobić kalsyfikacje
#   entropia histogramów (równych co 10%)
#   sprawdzić podział w bezwzględnym czasie (np co 0.5/0.4sec)
# =======================================

test = True


def prepare_data(df, path):
    # usuwam 3 sekundy zeby nie bylo nan
    df.drop(df.head(3000).index, inplace=True)
    df.drop(df.tail(3000).index, inplace=True)
    df.reset_index(drop=False, inplace=True)

    df['Hbeat'] = df['Hbeat'].astype(float, errors='raise')

    # df["Pierwsza_pochodna"] = np.gradient(df["Breath"].to_numpy())

    df["Pierwsza_pochodna"] = df["Breath"].diff() / 0.001

    peaks, _ = find_peaks(-df["Breath"], distance=900, prominence=0.05)

    # print(peaks)
    lst, poch_more3, poch_less3 = [],[],[]
    for peak in peaks:
        df_2 = df[peak: peak+200]
        lst.append(df_2['Pierwsza_pochodna'].max())
        if df_2[df_2['Pierwsza_pochodna'] > 0.2].empty:
            poch_more3.append(1)
        else:
            poch_less3.append(1)
    # print(lst)    
    print(len(poch_more3), len(poch_less3))
    print(sum(lst) / len(lst))
   

    df["Onset_breath"] = 0
    df["Hbeat_time"] = np.nan
    time1 = df["Time"].iloc[0]
    time2 = df["Time"].iloc[peaks[0]]

    # df = rank_hbeat_onset(df, time1, time2)

    for index, peak in enumerate(peaks):
        df.iloc[peak, df.columns.get_loc("Onset_breath")] = df["Breath"].iloc[peak]

        time1 = df["Time"].iloc[peak]
        if index == len(peaks) - 1:
            time2 = df["Time"].iloc[-1]
        else:
            time2 = df["Time"].iloc[peaks[index+1]]
        # df = rank_hbeat_onset(df,  time1, time2)
    return df


def rank_hbeat_onset(df, begin, end):
    ranks = []
    relative_time = np.subtract(end, begin)
    hbet = df.loc[(df["Hbeat"] != 0) & (df["Time"] > begin) &
                  (df["Time"] < end)]

    if relative_time % 0.5 != 0:
        time_range = int(relative_time / 0.5) + 1
    else:
        time_range = int(relative_time / 0.5)
    for i in range(time_range):
        ranks.append(0.5*(i+1))

    for index, row in hbet.iterrows():
        previous_time = 0
        for time in ranks:
            if ((row["Time"] - begin) <= time) and ((row["Time"] - begin) >
                                                    previous_time):
                df.loc[index, "Hbeat_time"] = time

            previous_time = time
    return df


def data_info(database):
    print(database.dtypes)
    print(database.describe())
    print(database.head())
    print(database.tail())


def show_sliced_graph(database, start, how_much):
    how_much = start + how_much
    sliced_df = database[(database["Time"] < how_much) & (database["Time"] >
                                                          start)]
    heart_beat = sliced_df[sliced_df["Hbeat"] != 0]
    onset_breath = sliced_df[sliced_df["Onset_breath"] != 0]

    fig, ax = plt.subplots()
    sns.lineplot(x="Time", y="Breath", data=sliced_df, ax=ax)
    sns.scatterplot(x="Time", y="Breath", data=heart_beat,
                    ax=ax, color="r")
    sns.scatterplot(x="Time", y="Breath", data=onset_breath, ax=ax,
                    color="g", s=200, marker="X")
    
    plt.show()
    return fig


def show_sliced_graph_movable(database):
    loop = True
    while loop is True:
        for i in range(10):
            try:
                start = int(input("Start(int): "))
                break
            except Exception:
                print("Invalid number")
                if i == 9:
                    start = 0
                continue

        for i in range(10):
            try:
                how_much = int(input("How long(int): "))
                break
            except Exception:
                print("Invalid number")
                if i == 9:
                    how_much = 10
                continue

        show_sliced_graph(database, start, how_much)
        plt.close()

        val = input("Koniec? (y/n): ").lower().strip()
        if val == "y":
            loop = False


def save_sliced_graph(graph, name):
    os.chdir("C:/Users/Mikołaj/Desktop/Licencjat/Pics")
    fig = graph
    fig.savefig(name + '.png')
    plt.close()


def isNaN(num):
    return num != num


def find_NaN(df, column):
    for index, row in df.iterrows():
        if isNaN(row[column]):
            print(index)


def frequency_mean(df_column, limit):
    asds = df_column.to_numpy()
    window_size = 10
    f, widmo = periodogram(asds, fs=1000)
    f = f[0:-window_size]
    widmo_mean = []
    for i in range(len(widmo) - window_size):
        widmo_mean.append(np.mean(widmo[i:i + window_size]))

    plt.plot(f, widmo_mean)
    plt.xlim(0, limit)
    plt.ylabel('Amplituda Widma Oddechu')
    plt.xlabel('Hz')
    return None


if __name__ == "__main__":
    path = "C:/Users/Mikołaj/Desktop/Licencjat/Dane/"
    txt_file_list = os.listdir(path)
    txt_file_single = txt_file_list[1]

    df = pd.read_csv(path + txt_file_single, delimiter="\t", header=None,
                      names=["Time", "Spasm", "Breath", 'Hbeat'])

    # df = df.iloc[:16000 + 16000]
    df = prepare_data(df, path)
#%%
    # for txt in txt_file_list:
    #     df = pd.read_csv(path + txt, delimiter="\t", header=None,
    #                       names=["Time", "Spasm", "Breath", 'Hbeat'])
    #     df = prepare_data(df, path)
    
    # fig, ax = plt.subplots()
    # sns.histplot(df, x="Hbeat_time",kde=True, ax=ax, binwidth = 0.5)
    # ax.set_title(f'Onset count: {np.count_nonzero(df["Onset_breath"]) - 1}')
    # fig.suptitle(txt_file_single)
    #     save_sliced_graph(fig, txt)

    # name = txt_file[1] + ".xlsx"
    # with pd.ExcelWriter("C:/Users/Mikołaj/Desktop/Licencjat/Dane_obrobione/" + name) as writer:
    #     df[:500000].to_excel(writer, sheet_name='first 500k')
    #     df[500000:1000000].to_excel(writer, sheet_nadddddddddddddddddddddd
    
    # frequency_mean(df['Breath'],0.5)
    # show_sliced_graph(df, 5, 12)
    # show_sliced_graph_movable(df)
    # save_sliced_graph(df, 3, 80, txt_file[-1])
    
    df2 = df.iloc[5000:16000]
    prog = 0.1

    fig, axs = plt.subplots(2)
    sns.lineplot(x="Time", y="Breath", data=df2, ax=axs[0])
    sns.lineplot(x="Time", y="Pierwsza_pochodna", data=df2, ax=axs[1])
    a = df2.loc[df["Onset_breath"] != 0, "Time"].to_numpy()
    granica = axs[1].axhline(y=prog, color = "g")
    for i in axs:
        for j in a:
            onset = i.axvline(j, color="red")
            onset_plus = i.axvline(j+0.4, color="black")
    fig.legend([granica,onset,onset_plus],[str(prog)+" granica","Onset","Onset + 400"])
