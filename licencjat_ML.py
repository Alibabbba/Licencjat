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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV


def ready_data(df):
    df = df.drop(["nazwa"], axis=1)
    X = df.drop(["klasa"], axis=1)
    df_class = df["klasa"].astype("str")
    label_encoder = preprocessing.LabelEncoder()
    Y = label_encoder.fit_transform(df_class)

    ros = RandomOverSampler()
    X, Y = ros.fit_resample(X, Y)

    scaler = preprocessing.StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_normalized = pd.DataFrame(preprocessing.normalize(X_scaled), columns=X.columns)

    return X_normalized, Y


def ready_classification(df):
    X_normalized, Y = ready_data(df)
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X_normalized, Y)
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
    param_grid_forest = {'bootstrap': [False, True], 'n_estimators': [2, 4, 6],
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
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()}")
    plt.show()
    randomforest.fit(X_train, Y_train)
    param_grid_forest = {'bootstrap': [False, True], 'n_estimators': [2, 4, 6],
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
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()}")
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
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()}")
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
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()}")
    plt.title("Confusion Matrix - Gaussian Naive")
    plt.show()
    return Y_pred, Y_test


def disp_confusion_matrix(Y_test, Y_pred, labels=None):
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()


def cross_val(classifier, df):
    df_data = df.drop(["nazwa"], axis=1)
    X = df_data.drop(["klasa"], axis=1)
    df_class = df_data["klasa"].astype("str")
    label_encoder = preprocessing.LabelEncoder()
    Y = label_encoder.fit_transform(df_class)

    classifier_score = np.sqrt(-1 * cross_val_score(
        classifier, X, Y, scoring="neg_mean_squared_error"))
    return classifier_score


if __name__ == "__main__":
    path = "C:/Users/Mikołaj/Desktop/Licencjat/"
    df = pd.read_csv(path + "out_df")

    labels = ["1A", "1B", "2A", "2B", "2C"]
    columns_names = ['frequency', 'oddech_na_min', 'R_do_2sec',
                     'R_na_min', 'entropia_oddech', 'entropia_onsety']
    X_train, X_test, Y_train, Y_test = ready_classification(df)

    ready_clustering(df)
    print("Random Forest Classifier")
    rand_pred, rand_true = random_forest_class(X_train, X_test, Y_train, Y_test)
    print("=============================")
    print("KNeighbors Classifier")
    kn_pred, kn_true = kneighborsClass(X_train, X_test, Y_train, Y_test)
    print("=============================")
    print("Decision Tree Classifier")
    tree_pred, tree_true = decisionTreeClass(X_train, X_test, Y_train, Y_test)
    print("=============================")
    print("Naive Bayes Classifier")
    gaus_pred, gaus_true = gaussian_naive(X_train, X_test, Y_train, Y_test)

    mcc_dic = {
        "Random Forest": matthews_corrcoef(rand_true, rand_pred),
        "KNeighbors": matthews_corrcoef(kn_true, kn_pred),
        "Decision Tree": matthews_corrcoef(tree_true, tree_pred),
        "Naive Bayes": matthews_corrcoef(gaus_true, gaus_pred)
        }

    print(mcc_dic)
# %%
    X_normalized, Y = ready_data(df)
    X_normalized["klasa"] = Y
    print(X_normalized)

    sns.scatterplot(x="R_na_min", y="oddech_na_min", data=X_normalized, hue="klasa")
