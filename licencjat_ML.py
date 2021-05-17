# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:08:12 2021

@author: Mikołaj
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, pipeline, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score


from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def ready_data(df):
    df = df.drop(["nazwa"], axis=1)
    X = df.drop(["klasa"], axis=1)

    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = preprocessing.normalize(X_scaled)
    X_normalized = pd.DataFrame(X_normalized)
    return X_normalized

def ready_classification(df):
    X_normalized = ready_data(df)
    df_class = df["klasa"].astype("str")
    label_encoder = preprocessing.LabelEncoder()
    Y = label_encoder.fit_transform(df_class)
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X_normalized, Y, random_state=0)
    return train_X, test_X, train_Y, test_Y


def ready_clustering(df):
    X_normalized = ready_data(df)
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_normalized)
    X_reduced = pd.DataFrame(X_reduced)
    X_reduced.columns = ['P1', 'P2', "P3"]
    print(abs(pca.components_))
    print("Ile wyjasnia", pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

    return X_reduced


def random_forest_class(X_train, X_test, Y_train, Y_test):
    randomforest = RandomForestClassifier()
    rfc = pipeline.Pipeline(steps=[('classifier', randomforest)])
    rfc.fit(X_train, Y_train)
    Y_pred_rfc = rfc.predict(X_test)

    disp_confusion_matrix(Y_test, Y_pred_rfc, labels=labels)
    plt.title("Confusion Matrix - RandomForest")
    print(f"Training: imput features = {X_train.shape}, output class = {Y_train.shape}")
    print(f"Testing: imput features = {X_test.shape}, output class = {Y_test.shape}")
    score = cross_val(randomforest, df)
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()}")
    plt.show()

    randomforest.fit(X_train, Y_train)
    plt.barh(columns_names, randomforest.feature_importances_)
    plt.title("Feature importance - RandomForest")
    plt.show()
    return Y_pred_rfc, Y_test


def kneighborsClass(X_train, X_test, Y_train, Y_test):
    knnClas = KNeighborsClassifier()
    knn_model = pipeline.Pipeline(steps=[('classifier', knnClas)])
    knn_model.fit(X_train, Y_train)
    Y_pred_knn = knn_model.predict(X_test)
    
    print("KNeighborsClass")
    disp_confusion_matrix(Y_test, Y_pred_knn, labels=labels)
    print(f"Training: imput features = {X_train.shape}, output class = {Y_train.shape}")
    print(f"Testing: imput features = {X_test.shape}, output class = {Y_test.shape}")
    score = cross_val(knnClas, df)
    print(f"Cross Validation(RMSE) scores: {score}, Avreges: {score.mean()}")
    plt.title("Confusion Matrix - KNeighbors")
    plt.show()
    return Y_pred_knn, Y_test


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
        classifier, X, Y, scoring="neg_mean_squared_error", cv=3))
    return classifier_score


if __name__ == "__main__":
    path = "C:/Users/Mikołaj/Desktop/Licencjat/"
    df = pd.read_csv(path + "out_df")
    brain_db = pd.read_excel(path + "BRAIN_database_UG_26_02_2021.xlsx")
    brain_picked = brain_db.loc[brain_db["Participant's code"].isin(df["nazwa"])]
    brain_rest = df.loc[~(df['nazwa'].isin(brain_picked["Participant's code"]))]
    brain_rest = [x[0:3] + " op " + x[4:7] for x in brain_rest['nazwa']]
    brain_rest = brain_db.loc[brain_db["Participant's code"].isin(brain_rest)]
    brain_picked = brain_picked.append(brain_rest).reset_index(drop=True)
    brain_picked["SEX"] = brain_picked['SEX'].fillna(0)

# %%
    X_clust = ready_clustering(df)
    db = DBSCAN(eps=1.5, min_samples=4)
    db.fit(X_clust)
    # print(db.labels_)
    labels = db.labels_

    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X_clust)
    distances, indices = neighbors_fit.kneighbors(X_clust)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()

    colours = {}
    colours[0] = 'r'
    colours[1] = 'g'
    colours[2] = 'b'
    colours[3] = 'c'
    colours[4] = 'y'
    colours[5] = 'm'
    colours[-1] = 'k'

    cvec = [colours[label] for label in labels]

    r = plt.scatter(X_clust['P1'], X_clust['P2'], color='r')
    g = plt.scatter(X_clust['P1'], X_clust['P2'], color='g')
    plt.close()


    markers = ["x", 'o']
    for i, marker in enumerate(markers):  
        plt.scatter(X_clust['P1'].loc[df['klasa'].str.startswith(str(i+1), na=False)],
                    X_clust['P2'].loc[df['klasa'].str.startswith(str(i+1), na=False)],
                    marker = marker)
    plt.legend((r, g), ('Label 0', 'Label 1'))
    plt.show()
    
    print(len(X_clust['P1'].loc[df['klasa'].str.startswith('1', na=False)]))
# %%
    df_test = df.iloc[0:2]
    labels = ["1A", "1B", "2A", "2B", "2C"]
    columns_names = ['frequency', 'oddech_na_min', 'R_do_2sec', 'R_na_min', 'entropia_oddech', 'entropia_onsety']
    X_train, X_test, Y_train, Y_test = ready_classification(df)

    print("Random Forest")
    random_forest_class(X_train, X_test, Y_train, Y_test)
    print("=============================")
    print("KNeighborsClass")
    kneighborsClass(X_train, X_test, Y_train, Y_test)
# %%

    X_normalized = ready_data(df)
    print(X_normalized)
    
    sns.scatter()

