# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:08:12 2021

@author: Mikołaj
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from scipy import stats
import scikit_posthocs as sp
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter

# =============================================================================
# 0 - 1A, 1 - 1B, 2 - 2A, 3 - 2B, 4 - 2C
# =============================================================================


def ready_data(df):
    df = df.drop(["nazwa"], axis=1)
    X = df.drop(["klasa"], axis=1)
    df_class = df["klasa"].astype("str")
    label_encoder = preprocessing.LabelEncoder()
    Y = label_encoder.fit_transform(df_class)
    # --------------------------
    print(Counter(Y))
    sampling_strategy = {0: 11, 1: 7, 2: 18, 3: 16, 4: 7}
    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    X, Y = ros.fit_resample(X, Y)
    # --------------------------
    print(Counter(Y))
    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)
    print(Counter(Y))

    scaler = preprocessing.StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_normalized = pd.DataFrame(preprocessing.normalize(X_scaled), columns=X.columns)

    return X_normalized, Y


def ready_classification(df):
    X_normalized, Y = ready_data(df)
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X_normalized, Y, stratify=Y, random_state=0)
    return train_X, test_X, train_Y, test_Y


def ready_clustering(df):
    X_normalized, _ = ready_data(df)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_normalized)
    X_reduced = pd.DataFrame(X_reduced)
    X_reduced.columns = ['P1', 'P2']
    print(abs(pca.components_))
    print("Ile wyjasnia", pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

    return X_reduced


def random_forest_class(X_train, X_test, Y_train, Y_test):
    randomforest = RandomForestClassifier()
    randomforest.fit(X_train, Y_train)
    param_grid_forest = {'bootstrap': [False, True], 'n_estimators': [100, 500, 1000],
                         'max_features': [2, 3, 4], 'max_leaf_nodes': [5, 50, 500, 5000]}
    # Y_pred_rfc = rfc.predict(X_test)

    grid = GridSearchCV(randomforest, param_grid_forest,
                        cv=10, scoring='neg_mean_squared_error')
    grid.fit(X_train, Y_train)
    print(f"Best parametrs: {grid.best_estimator_}")
    grid_predictions = grid.predict(X_test)

    disp_confusion_matrix(Y_test, grid_predictions, labels=labels)
    plt.title("Confusion Matrix - RandomForest")
    print(f"Training: imput features = {X_train.shape}, output class = {Y_train.shape}")
    print(f"Testing: imput features = {X_test.shape}, output class = {Y_test.shape}")
    score = cross_val(randomforest, df)
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()} standard deviation: {score.std()}")
    plt.show()
    randomforest.fit(X_train, Y_train)
    param_grid_forest = {'bootstrap': [False, True], 'n_estimators': [100, 500, 1000],
                         'max_features': [2, 3, 4], 'max_leaf_nodes': [5, 50, 500, 5000]}
    plt.barh(columns_names, randomforest.feature_importances_)
    plt.title("Feature importance - RandomForest")
    plt.show()
    return grid_predictions, Y_test


def kneighborsClass(X_train, X_test, Y_train, Y_test):
    knnClas = KNeighborsClassifier()
    knnClas.fit(X_train, Y_train)
    y_pred = knnClas.predict(X_test)

    disp_confusion_matrix(Y_test, y_pred, labels=labels)
    print(f"Training: imput features = {X_train.shape}, output class = {Y_train.shape}")
    print(f"Testing: imput features = {X_test.shape}, output class = {Y_test.shape}")
    score = cross_val(knnClas, df)
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()} standard deviation: {score.std()}")
    plt.title("Confusion Matrix - KNeighbors")
    plt.show()
    return y_pred, Y_test


def decisionTreeClass(X_train, X_test, Y_train, Y_test, tree_plot=False):
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree = decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)

    disp_confusion_matrix(Y_test, Y_pred, labels=labels)
    print(f"Training: imput features = {X_train.shape}, output class = {Y_train.shape}")
    print(f"Testing: imput features = {X_test.shape}, output class = {Y_test.shape}")
    score = cross_val(decision_tree, df)
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()} standard deviation: {score.std()}")
    plt.title("Confusion Matrix - DecisionTree")
    plt.show()
    if tree_plot is True:
        plt.figure(figsize=(20, 15))
        plot_tree(decision_tree, feature_names=X_train.columns, fontsize=8)
        os.chdir("C:/Users/Mikołaj/Desktop/Licencjat/Pics")
        plt.savefig('decision_tree.pdf')
    return Y_pred, Y_test


def gaussian_naive(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()
    Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    disp_confusion_matrix(Y_test, Y_pred, labels=labels)
    print(f"Training: imput features = {X_train.shape}, output class = {Y_train.shape}")
    print(f"Testing: imput features = {X_test.shape}, output class = {Y_test.shape}")
    score = cross_val(gnb, df)
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()} standard deviation: {score.std()}")
    plt.title("Confusion Matrix - Gaussian Naive")
    plt.show()
    return Y_pred, Y_test


def disp_confusion_matrix(Y_test, Y_pred, labels=None):
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    return cm


def cross_val(classifier, df):
    df_data = df.drop(["nazwa"], axis=1)
    X = df_data.drop(["klasa"], axis=1)
    df_class = df_data["klasa"].astype("str")
    label_encoder = preprocessing.LabelEncoder()
    Y = label_encoder.fit_transform(df_class)

    classifier_score = np.sqrt(-1 * cross_val_score(
        classifier, X, Y, scoring="neg_mean_squared_error"))
    return classifier_score


def f1_coin(true, pred):
    for bol in [true, pred]:
        for i, x in enumerate(bol):
            bol[i] = str(x)
    mcm = multilabel_confusion_matrix(true, pred)
    out_dict = {}
    for label in set(true):
        tn, fp, fn, tp = mcm[label].ravel()
        q = (tp+fn)/(tp+tn+fp+fn)
        f1_coin = round(2*q/(q+1), 2)
        f1 = 2/(1/(tp/(tp+fp)) + 1/(tp/(tp+fn)))
        f1_norm = round((f1-f1_coin)/(1-f1_coin), 2)
        out_dict[str(label)] = f"f1_norm: {f1_norm}, f1 los: {f1_coin}, f1: {round(f1,2)}"

    return out_dict


if __name__ == "__main__":
    path = "C:/Users/Mikołaj/Desktop/Licencjat/"
    df = pd.read_csv(path + "out_df_poprawa")
    # df = pd.read_csv(path + "out_df_poprawa_3klasy")
    #%%

    labels = df["klasa"].unique().tolist()
    print(labels)
    columns_names = ['frequency', 'oddech_na_min', 'R_do_2sec', 
                     'R_na_min', 'entropia_oddech', 'entropia_onsety']
    X_train, X_test, Y_train, Y_test = ready_classification(df)


    mcc_dicc = {}
    for classifier in [random_forest_class, kneighborsClass, decisionTreeClass, gaussian_naive]:
        print(classifier.__name__)
        pred, true = classifier(X_train, X_test, Y_train, Y_test)
        print(pred, true)
        print(classification_report(true, pred, target_names=labels))
        print(f1_coin(true, pred))
        mcc_dicc[classifier.__name__] = round(matthews_corrcoef(true, pred), 3)
        print("=============================")

    print(mcc_dicc)


    # %%
    # print(stats.kruskal(df["entropia_hbeat"], df["entropia_oddech"], df["R_do_2sec"],
                        # df["R_na_min"]))

    klasy = []
    for klasa in df["klasa"].unique():
        a = df.loc[df["klasa"] == klasa]
        a = a.drop(["nazwa", "klasa"], axis=1)
        klasy.append(a)
    for cecha in klasy[0].columns:
        print(cecha)
        a1 = klasy[0][cecha].to_numpy()
        a2 = klasy[1][cecha].to_numpy()
        a3 = klasy[2][cecha].to_numpy()
        a4 = klasy[3][cecha].to_numpy()
        a5 = klasy[4][cecha].to_numpy()
        stat, pval = stats.kruskal(a1, a2, a3, a4, a5)
        print(f'Statistic: {stat} p-value: {pval}')
        if pval < 0.1:            
            grupa = [a1, a2, a3, a4, a5]
            print(sp.posthoc_dunn(grupa))
            print(df["klasa"].unique())


    # X_normalized, Y = ready_data(df)
    # X_normalized["klasa"] = Y
    # print(X_normalized)
    # sns.scatterplot(x="R_na_min", y="oddech_na_min", data=X_normalized, hue="klasa")
