# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:45:20 2018

@author: Srinivasan Thangamani
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
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


# Function to calculate age from YOB
def age_calc_process(df, lower_limit=10, upper_limit=75):
    df = df.copy(deep=True)
    current_year = datetime.datetime.now().year
    df['age'] = current_year - df['yob']
    df.drop(['yob'], axis=1, inplace=True)
    df = df[(df['age']>=lower_limit) & (df['age']<=upper_limit)]
    return df

# Function to drop columns
def drop_cols(df, cols):
    df = df.copy(deep=True)
    df.drop(cols, axis=1, inplace=True)
    return df

# Function to drop NA rows
def drop_rows(df, cols):
    df = df.copy(deep=True)
    df.dropna(axis=0, inplace=True,subset=cols)
    return df

# Function to encode the columns
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


# Function to select the features based on p-Value
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


# Function to perform PCA - Dimensionality Reduction
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


# Function to select features using multivariate analysis - Regression
def multivariate_feature_selection(X, Y):
    available_cols = list(X.columns)
    models = {'random_forest':RandomForestRegressor(max_features='log2',n_jobs=-1, random_state=2017)}
    clf = models.get('random_forest')
    selector = RFECV(estimator=clf, step=0.1, cv=5, n_jobs=-1,
                             scoring='r2')
    selector = selector.fit(X, Y)
    no_of_features = selector.n_features_
    feature_index = selector.support_.tolist()
    selected_cols = list(itertools.compress(available_cols, feature_index))
    return  {"column": selected_cols, "n_features": no_of_features}


# Function to select features using multivariate analysis - Classification
def feature_selection_classification(X, Y):
    available_cols = X.columns
    clf = RandomForestClassifier(max_features='log2',
                                             n_jobs=-1, random_state=2017,
                                             class_weight='balanced')
    selector = RFECV(estimator=clf, step=0.1, cv=5, n_jobs=-1,
                             scoring='accuracy')
    selector = selector.fit(X, Y)
    no_of_features = selector.n_features_
    feature_index = selector.support_.tolist()
    selected_cols = list(itertools.compress(available_cols, feature_index))
    results = {"column": selected_cols, "n_features": no_of_features}
    return results


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
m = encode_cols(df=m, cols=['gender','device_category'])

X = m[m.columns.difference(['age','ifa'])]
Y = m.loc[:, ('age')]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.15)

results_uni = select_features(x_train=X_train, y_train=Y_train, pvalue=0.05)
results_uni_pd = pd.DataFrame(results_uni)
results_uni_pd.to_csv('results_uni.csv', index= False)

pca_results = reduce_dimension(x_train=X_train)
pca_data = pca_results.get('PCA')
pca_results.get('explained_variance')
top_variance = pca_data[:,:2]

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(top_variance, Y_train)
y_pred_pca = regr.predict(top_variance)
print(mean_squared_error(Y_train, y_pred_pca, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_train, y_pred_pca))
# PCA Results are not satisfactory

regr_uni = RandomForestRegressor(max_depth=2, random_state=0, criterion ='mae', max_features ='log2', n_jobs=-1)
regr_uni.fit(X_train[results_uni.get('column')], Y_train)
y_pred_uni = regr_uni.predict(X_train[results_uni.get('column')])
print(mean_squared_error(Y_train, y_pred_uni, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_train, y_pred_uni))
joblib.dump(regr_uni, 'RF_Univariate_Regression.pkl')


results_rfecv = multivariate_feature_selection(X=X_train , Y=Y_train)
results_rfecv_pd = pd.DataFrame(results_rfecv)
results_rfecv_pd.to_csv('results_rfecv_multi.csv', index= False)


regr_multi = RandomForestRegressor(max_depth=5, random_state=0, criterion ='mae', max_features ='log2', n_jobs=-1)
regr_multi.fit(X_train[results_rfecv.get('column')], Y_train)
y_pred_multi = regr_multi.predict(X_train[results_rfecv.get('column')])
print(mean_squared_error(Y_train, y_pred_multi, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_train, y_pred_multi))
joblib.dump(regr_multi, 'RF_multivariate_Regression.pkl')
loaded_model = joblib.load('RF_multivariate_Regression.pkl')
pd.DataFrame({"Columns":results_rfecv.get('column'),"Importance":loaded_model.feature_importances_}).to_csv(
    'feature_importance_regression.csv',index=False)

y_pred_multi_test = loaded_model.predict(X_test[results_rfecv.get('column')])
print(mean_squared_error(Y_test, y_pred_multi_test, multioutput='raw_values'))
print(mean_absolute_percentage_error(Y_test, y_pred_multi_test))

data_classification = age_calc_process(df=result,lower_limit=18, upper_limit=85)
# Drop the columns which are not required for processing
cols_to_remove = ['first_seen', 'last_seen', 'platform','total_conn_brq']
data_classification = drop_cols(df=data_classification, cols=cols_to_remove)
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

data_classification = drop_rows(df=data_classification, cols=rm_na_cols)
data_classification = encode_cols(df=data_classification, cols=['gender','device_category'])

# Separate ages into bins
data_classification['age_range'] = pd.cut(data_classification['age'],[17,24,34,44,54,85],labels=[1,2,3,4,5])

# Save the data as csv
data_classification.to_csv('data_classification_85.csv', index=False)

# Extract dependent and independent variables
X_Classification = data_classification[data_classification.columns.difference(['age','ifa','age_range'])]
Y_Classification = data_classification.loc[:, ('age_range')]

# Feature selection

features_classification = feature_selection_classification(X=X_Classification, Y=Y_Classification)
features_classification_pd = pd.DataFrame(features_classification)
features_classification_pd.to_csv('features_classification.csv',index=False)


X_class_train, X_class_test, Y_class_train, Y_class_test = train_test_split(X_Classification, Y_Classification,
                                                          stratify=Y_Classification, test_size=0.15)

# Classification Using Random Forest Classifier

param_grid = {}
rand_cv = {}
best_score = 0
models ={}

all_models = {'RandomForest': RandomForestClassifier(n_jobs=-1),
              'ExtraTreesClassifier': ExtraTreesClassifier(n_jobs=-1)}

param_grid['RandomForest'] = {'max_features': ['auto', 'log2', None], 'class_weight': ['balanced']}
param_grid['ExtraTreesClassifier'] = {'class_weight': ['balanced'], 'max_features': ['auto', 'log2', 'sqrt', None]}

for name, model in all_models.items():
    rand_cv[name] = RandomizedSearchCV(model,
                                 param_grid[name],
                                 cv=5,
                                 scoring='accuracy',
                                 n_iter=3,
                                 random_state=2017,
                                 refit='accuracy',
                                 return_train_score=True,
                                 n_jobs=3)
    # Fit the model
    rand_cv[name].fit(X_class_train[features_classification.get('column')], Y_class_train)
    if rand_cv[name].best_score_ > best_score:
        best_score = rand_cv[name].best_score_
        best_model_name = name
    models[name] = rand_cv[name].best_estimator_
    joblib.dump(models[best_model_name], best_model_name+'.pkl')

# Training data prediction

loaded_model = joblib.load('RandomForest.pkl')

rf_class = loaded_model.predict(X_class_train[features_classification.get('column')])
print('Training Data Accuracy:', accuracy_score(Y_class_train, rf_class)*100)

rf_class_test = loaded_model.predict(X_class_test[features_classification.get('column')])
print('Test Data Accuracy:',accuracy_score(Y_class_test, rf_class_test)*100)

feature_importance_pd = pd.DataFrame({"Feature":features_classification.get('column'),
                         "Importance":loaded_model.feature_importances_})
feature_importance_pd.to_csv('feature_importance_classification.csv',index=False)


loaded_model.predict_proba(X_class_test[features_classification.get('column')])