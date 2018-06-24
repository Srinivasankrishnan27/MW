# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:45:20 2018

@author: P1311415
"""

import pandas as pd
import datetime
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
import itertools
import numpy as np
from sklearn.decomposition import PCA

import os
from sklearn.ensemble import RandomForestRegressor

def age_calc_process(df, lower_limit=10, upper_limit=75):
    df = df.copy(deep=True)
    current_year = datetime.datetime.now().year
    df['age'] = current_year - df['yob']
    df.drop(['yob'], axis=1, inplace=True)
    df = df[(df['age']>=lower_limit) & (df['age']<=upper_limit)]
    return df


def drop_cols(df, cols):
    df = df.copy(deep=True)
    df.drop(cols, axis=1, inplace=True)
    return df


def drop_rows(df, cols):
    df = df.copy(deep=True)
    df.dropna(axis=0, inplace=True,subset=cols)
    return df


def encode_cols(df, cols):
    """
    Encode the categorical columns.

    Args:
        df : Data frame
        cols (list): list of columns to be encoded
    Returns:
        df_to_encode: Data frame with the encoded columns.
    """
    df_to_encode = df.copy(deep=True)
    for i in cols:
        df_to_encode[i] = df_to_encode[i].astype(dtype='category')
        df_to_encode = pd.get_dummies(df_to_encode, columns=[i],
                                      prefix_sep='_', dummy_na=False)
    return df_to_encode


def select_features(x_train, y_train, pvalue=0.05):
    """Select columns based on feature importance.

    Args:
        x_train (DataFrame): Independent variables
        y_train (list): Target column
        pvalue (float): Threshold to select the features, default(0.05)
    Returns:
        results (dict): {"column": selected_cols, "n_features": no_of_features}

    """
    available_cols = list(x_train.columns)
    f_selector = feature_selection.f_classif(X=x_train, y=y_train)
    p_value = f_selector[1]
    print(p_value)
    index = np.argwhere(p_value < pvalue).tolist()
    pvalue_index = list(itertools.chain.from_iterable(index))
    selected_cols = list(map(lambda x: available_cols[x], pvalue_index))
    no_of_features = len(selected_cols)
    results = {"column": selected_cols, "n_features": no_of_features,
               "model": "Anova"}
    return results


def reduce_dimension(x_train):
    """Reduce dimension of the data set.

    Args:
        x_train (DataFrame): Independent variables
    Returns:
        results (dict): {"method": method name (PCA), "explained_variance": explained_variance, "PCA": x_pca}

    """
    pca = PCA(svd_solver='auto', random_state=2017)
    x_pca = pca.fit_transform(x_train)
    explained_variance = pca.explained_variance_ratio_.tolist()
    results = {'method': 'PCA', 'explained_variance': explained_variance, 'PCA': x_pca}
    return results


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Read the source file
data_dict = pd.read_excel('data_dictionary.xlsx',sheet_name='features')
brq = pd.read_csv('brq.csv')
user_app_details = pd.read_csv('user_app_details.csv')

# Merge user_app_details and brq
result = pd.merge(user_app_details, brq,left_on='ifa', right_on='ifa')

# Calculate the age and drop the yob column
m = age_calc_process(df=result)
# Drop the columns which are not required for processing
cols_to_remove = ['first_seen', 'last_seen', 'platform','total_conn_brq']
m = drop_cols(df=m, cols=cols_to_remove)
# Drop rows which has NA's cannot be imputed

rm_na_cols=['device_category','gender','CELLULAR','WIFI','rog_m',
                          'ANDROID-ART_AND_DESIGN','ANDROID-AUTO_AND_VEHICLES', 'ANDROID-BEAUTY',
                          'ANDROID-BOOKS_AND_REFERENCE', 'ANDROID-BUSINESS', 'ANDROID-COMICS',
                          'ANDROID-COMMUNICATION', 'ANDROID-DATING', 'ANDROID-EDUCATION',
                          'ANDROID-ENTERTAINMENT', 'ANDROID-EVENTS', 'ANDROID-FINANCE',
                          'ANDROID-FOOD_AND_DRINK', 'ANDROID-GAME_ACTION','ANDROID-GAME_ADVENTURE',
                          'ANDROID-GAME_ARCADE', 'ANDROID-GAME_BOARD','ANDROID-GAME_CARD',
                          'ANDROID-GAME_CASINO', 'ANDROID-GAME_CASUAL','ANDROID-GAME_EDUCATIONAL',
                          'ANDROID-GAME_MUSIC', 'ANDROID-GAME_PUZZLE','ANDROID-GAME_RACING',
                          'ANDROID-GAME_ROLE_PLAYING','ANDROID-GAME_SIMULATION','ANDROID-GAME_SPORTS',
                          'ANDROID-GAME_STRATEGY', 'ANDROID-GAME_TRIVIA', 'ANDROID-GAME_WORD',
                          'ANDROID-HEALTH_AND_FITNESS', 'ANDROID-HOUSE_AND_HOME','ANDROID-LIBRARIES_AND_DEMO',
                          'ANDROID-LIFESTYLE','ANDROID-MAPS_AND_NAVIGATION', 'ANDROID-MEDICAL',
                          'ANDROID-MUSIC_AND_AUDIO', 'ANDROID-NEWS_AND_MAGAZINES','ANDROID-PARENTING',
                          'ANDROID-PERSONALIZATION', 'ANDROID-PHOTOGRAPHY','ANDROID-PRODUCTIVITY',
                          'ANDROID-SHOPPING', 'ANDROID-SOCIAL','ANDROID-SPORTS', 'ANDROID-TOOLS',
                          'ANDROID-TRAVEL_AND_LOCAL','ANDROID-VIDEO_PLAYERS','ANDROID-WEATHER']

m = drop_rows(df=m, cols=rm_na_cols)
print(m.isna().sum())
print(len(m))

m = encode_cols(df=m, cols=['gender','device_category'])

X = m[m.columns.difference(['age','ifa'])]
Y = m.loc[:, ('age')]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.15)

results_uni = select_features(x_train=X_train, y_train=Y_train, pvalue=0.05)
results_uni.get('column')

pca_results = reduce_dimension(x_train=X_train)
pca_data = pca_results.get('PCA')
pca_results.get('explained_variance')
top_variance = pca_data[:,:2]

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(top_variance, Y_train)
y_pred_pca = regr.predict(top_variance)
print(mean_squared_error(Y_train, y_pred_pca, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_train, y_pred_pca))


regr_uni = RandomForestRegressor(max_depth=2, random_state=0)
regr_uni.fit(X_train[results_uni.get('column')], Y_train)
y_pred_uni = regr_uni.predict(X_train[results_uni.get('column')])
print(mean_squared_error(Y_train, y_pred_uni, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_train, y_pred_uni))

np.savetxt('sample.csv',regr_uni.feature_importances_,delimiter=',')

available_cols = list(X_train.columns)
models = {'random_forest':RandomForestRegressor(max_features='log2',n_jobs=-1, random_state=2017)}
clf = models.get('random_forest')
selector = RFECV(estimator=clf, step=0.1, cv=5, n_jobs=-1,
                         scoring='r2')
selector = selector.fit(X_train, Y_train)
no_of_features = selector.n_features_
feature_index = selector.support_.tolist()
selected_cols = list(itertools.compress(available_cols, feature_index))
results_rfecv = {"column": selected_cols, "n_features": no_of_features}



regr_multi = RandomForestRegressor(max_depth=2, random_state=0)
regr_multi.fit(X_train[results_rfecv.get('column')], Y_train)
y_pred_multi = regr_multi.predict(X_train[results_rfecv.get('column')])
print(mean_squared_error(Y_train, y_pred_multi, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_train, y_pred_multi))


m[results_rfecv.get('column')].to_csv('processed.csv', index=False)
Y.to_csv('processed_target.csv', index=False)