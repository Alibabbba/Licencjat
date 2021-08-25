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
import math

from scipy.signal import periodogram, find_peaks
from scipy.stats import entropy

# =======================================

#   cechy charakterystyczne: entropia, ilosc oddechow na minutę, frequency,
#                            ilosc R (maksymalna frequency)

# =======================================


def prepare_data(df):
    # usuwam 3 sekundy zeby nie bylo nan
    df.drop(df.head(3000).index, inplace=True)
    df.drop(df.tail(3000).index, inplace=True)
    df.reset_index(drop=False, inplace=True)

    df['Hbeat'] = df['Hbeat'].astype(float)
    df["Pierwsza_pochodna"] = df["Breath"].diff() / 0.001

    peaks, _ = find_peaks(-df["Breath"], distance=900, prominence=0.05)

    for index, peak in enumerate(peaks):
        df_2 = df[peak: peak+400]
        if not df_2[df_2["Pierwsza_pochodna"] > 0.3].empty:
            peaks[index] = df_2.loc[
                df_2["Pierwsza_pochodna"] > 0.3, "index"].iloc[0] - 3000
        elif not df_2[df_2["Pierwsza_pochodna"] > 0.2].empty:
            peaks[index] = df_2.loc[
                df_2["Pierwsza_pochodna"] > 0.2, "index"].iloc[0] - 3000
        elif not df_2[df_2["Pierwsza_pochodna"] > 0.1].empty:
            peaks[index] = df_2.loc[
                df_2["Pierwsza_pochodna"] > 0.1, "index"].iloc[0] - 3000

    df["Onset_breath"] = 0
    df["Hbeat_time"] = np.nan

    for index, peak in enumerate(peaks):
        if index == 0:
            time1 = df["Time"].iloc[0]
            time2 = df["Time"].iloc[peaks[0]]
            df = rank_hbeat_onset(df, time1, time2)

        # nadaje wartosc Breath o indeksie peak, indeksowi peak w kolumnie Onset_breath
        df.iloc[peak, df.columns.get_loc("Onset_breath")] = df["Breath"].iloc[peak]
        time1 = df["Time"].iloc[peak]
        if index == len(peaks) - 1:
            time2 = df["Time"].iloc[-1]
        else:
            time2 = df["Time"].iloc[peaks[index+1]]
        df = rank_hbeat_onset(df,  time1, time2)
    return df


def rank_hbeat_onset(df, begin, end):
    ranks = []
    relative_time = np.subtract(end, begin)
    hbet = df.loc[(df["Spasm"] != 0) & (df["Time"] > begin) &
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
            if ((row["Time"] - begin) <= time) and ((row["Time"] - begin) > previous_time):
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
    heart_beat = sliced_df[sliced_df["Spasm"] != 0]
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
    fig.suptitle(name)
    fig.savefig(name + '.png')
    plt.close()


def isNaN(num):
    return num != num


def find_NaN(df, column):
    for index, row in df.iterrows():
        if isNaN(row[column]):
            print(index)


def onset_count(df):
    hist_size = len(pd.unique(df['Hbeat_time'])) - 1
    bars_height = df.Hbeat_time.dropna().to_numpy()

    fig, ax = plt.subplots()
    # sns.hisplot(data=df, x="Hbeat_time", ax=ax, clip=(0.5, hist_size * 0.5))
    bar_values = ax.hist(bars_height, bins=hist_size,
                         density=True, histtype='bar', ec='black')[0]
    ent = -(bar_values*np.log(bar_values))
    ent = sum([0 if math.isnan(i) else i for i in ent])
    ax.set_title(f'Ada_1A_007Sel \nIlość wdechów: {np.count_nonzero(df["Onset_breath"]) - 1}, Entropia: {round(ent, 4)}')
    ax.set_ylabel("Szansa na spazm serca")
    ax.set_xlabel("Sekundy")
    return fig


def onset_count_entropy(df_Hbeat_time):
    hist_size = len(pd.unique(df_Hbeat_time)) - 1
    bars_height = df_Hbeat_time.dropna().to_numpy()
    hist = np.histogram(bars_height, bins=hist_size, density=True)[0]
    with np.errstate(divide='raise'):
        try:
            ent = entropy(hist)
        except FloatingPointError:
            hist = [1 if i == 0 else i for i in hist]
            ent = sum(-(hist*np.log(hist)))
    return ent


def plot_deriv_breath_onset(df, szukana_granica, min_sec, max_sec):
    df2 = df.iloc[(min_sec-3) * 1000:(max_sec-3) * 1000]

    prog = szukana_granica
    fig, axs = plt.subplots(2)
    sns.lineplot(x="Time", y="Breath", data=df2, ax=axs[0])
    sns.lineplot(x="Time", y="Pierwsza_pochodna", data=df2, ax=axs[1])
    a = df2.loc[df["Onset_breath"] != 0, "Time"].to_numpy()
    granica = axs[1].axhline(y=prog, color="g")
    for i in axs:
        for j in a:
            onset = i.axvline(j, color="red")
            onset_plus = i.axvline(j+0.4, color="black")
#   fig.legend([granica, onset, onset_plus], [str(prog)+" granica", "Onset", "Onset + 400"])
    return fig


def frequency_mean(df_column, limit, plot=True):
    asds = df_column.to_numpy()
    window_size = 50
    f, widmo = periodogram(asds, fs=1000)
    f = f[0:1000-window_size]
    widmo = widmo[0:1000]
    widmo_mean = []
    for i in range(len(widmo) - window_size):
        widmo_mean.append(np.mean(widmo[i:i + window_size]))
    if plot is True:
        fig = plt.figure()
        plt.plot(f, widmo_mean)
        plt.xlim(0, limit)
        plt.ylabel('Amplituda Widma Oddechu')
        plt.xlabel('Hz')
        plt.title(f"Max = {f[widmo_mean.index(max(widmo_mean))]}")
        return widmo_mean, f, fig
    else:
        return widmo_mean, f


def prepare_outcome_df(path):
    outcome_df = pd.DataFrame(
        columns=["nazwa", "entropia_hbeat", "entropia_oddech",
                 "frequency", "oddech_na_min", "R_do_2sec", "R_na_min", "klasa"])
    txt_file_list = os.listdir(path)
    for index, file_name in enumerate(txt_file_list):
        in_df = pd.read_csv(path + file_name, delimiter="\t", header=None,
                            names=["Time", "Spasm", "Breath", 'Hbeat'])
        in_df = prepare_data(in_df)
        widmo, f = frequency_mean(in_df['Breath'], 0.5, plot=False)
        entropia_oddech_list = []
        window_size = int(len(widmo)/11)
        for i in range(0, len(widmo)-window_size, window_size):
            entropia_oddech_list.append(np.mean(widmo[i:i+window_size]))

        entropia_onsety = onset_count_entropy(in_df["Hbeat_time"])
        entropia_oddech = entropy(entropia_oddech_list)
        frequency = f[widmo.index(max(widmo))]
        oddechy_na_min = np.count_nonzero(in_df["Onset_breath"])
        R_do_2sec = in_df.loc[(in_df["Spasm"] != 0) & (in_df["Hbeat_time"] < 2), "Hbeat_time"].count()
        R_do_2sec = R_do_2sec / oddechy_na_min
        R_na_min = in_df.loc[in_df["Spasm"] != 0, "Hbeat"].size / 20
        oddechy_na_min = oddechy_na_min / 20
        klasa = file_name[4:6]

        if int(file_name[4]) == 1:
            name = "I" + file_name[5] + " " + file_name[7:10]
            # klasa = '1'
        else:
            name = "II" + file_name[5] + " " + file_name[7:10]
            # klasa = "2"

        outcome_df = outcome_df.append({
            "nazwa": name,
            "entropia_hbeat": entropia_onsety,
            "entropia_oddech": entropia_oddech,
            "frequency": frequency,
            "oddech_na_min": oddechy_na_min,
            "R_do_2sec": R_do_2sec,
            "R_na_min": R_na_min,
            "klasa": klasa
              }, ignore_index=True)
        print(f"Done {index+1} out of {len(txt_file_list)}")
    return outcome_df


if __name__ == "__main__":
    path = "C:/Users/Mikołaj/Desktop/Licencjat/Dane_3_klasy/"
    out_path = "C:/Users/Mikołaj/Desktop/Licencjat/"
    txt_file_list = os.listdir(path)
    txt_file_single = txt_file_list[0]

    df = pd.read_csv(path + txt_file_single, delimiter="\t", header=None,
                      names=["Time", "Spasm", "Breath", 'Hbeat'])

    # df = df.iloc[:16000 + 16000]
    df = prepare_data(df)

    # widmo, f = frequency_mean(df['Breath'], 0.5, plot=False)

    # lis = []
    # window_size = int(len(widmo)/11)
    # for i in range(0,len(widmo)-window_size , window_size):
    #     lis.append(np.mean(widmo[i:i+window_size]))
    # print(sum(-(lis*np.log(lis))))

    # print(onset_count_entropy(df["Hbeat_time"]))
    # onset_count(df)
    # frequency_mean(df['Breath'],0.5)
    # widmo, f, fig = frequency_mean(df['Breath'], 0.5, plot=True)
    # frequency = f[widmo.index(max(widmo))]
    # print(frequency)

    # show_sliced_graph(df, 5, 12)
    # show_sliced_graph_movable(df)
    # save_sliced_graph(df, 3, 80, txt_file[-1])
    # data_info(df)
    # fig = plot_deriv_breath_onset(df, 0.1, 11, 22)
    # plt.title("as")
    # print(onset_count_entropy(df["Hbeat_time"]))

    # for txt in txt_file_list:
    #     df = pd.read_csv(path + txt, delimiter="\t", header=None,
    #                       names=["Time", "Spasm", "Breath", 'Hbeat'])
    #     df = prepare_data(df)
        # plots = plot_deriv_breath_onset(df, 0.1, 8, 19)
        # plots
    #     widmo, f, fig = frequency_mean(df['Breath'], 0.5, plot=True)
    #     frequency = f[widmo.index(max(widmo))]

    #     save_sliced_graph(fig, txt)

    # test_path = "C:/Users/Mikołaj/Desktop/Licencjat"
    out_df = prepare_outcome_df(path)
    out_df.to_csv(path_or_buf=out_path + "out_df_poprawa_3klasy", index=False)

    print(out_df)
    # out_df.to_excel(out_path + "out_df_excel2.xlsx", index=False)
