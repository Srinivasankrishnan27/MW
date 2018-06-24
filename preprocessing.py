# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:45:20 2018

@author: P1311415
"""

import pandas as pd
import datetime
import os
from sklearn.ensemble import RandomForestRegressor


# Read the source file
data_dict = pd.read_excel('data_dictionary.xlsx',sheet_name='features')
brq = pd.read_csv('brq.csv')
user_app_details = pd.read_csv('user_app_details.csv')

# Merge user_app_details and brq 
result = pd.merge(user_app_details, brq,left_on='ifa', right_on='ifa')

# Filter the cols for further processing
to_process = result.loc[:,('ifa','num_days', 'brq','distinctasn', 'gender', 'yob', 'distinctloc',
              'home', 'datepart_other', 'work', 'total_loc_brq', 'CELLULAR', 
              'WIFI','rog_m', 'ANDROID-ART_AND_DESIGN','ANDROID-AUTO_AND_VEHICLES', 
              'ANDROID-BEAUTY','ANDROID-BOOKS_AND_REFERENCE', 'ANDROID-BUSINESS', 
              'ANDROID-COMICS','ANDROID-COMMUNICATION', 'ANDROID-DATING', 
              'ANDROID-EDUCATION','ANDROID-ENTERTAINMENT', 'ANDROID-EVENTS', 
              'ANDROID-FINANCE','ANDROID-FOOD_AND_DRINK', 'ANDROID-GAME_ACTION',
              'ANDROID-GAME_ADVENTURE', 'ANDROID-GAME_ARCADE', 'ANDROID-GAME_BOARD',
              'ANDROID-GAME_CARD', 'ANDROID-GAME_CASINO', 'ANDROID-GAME_CASUAL',
              'ANDROID-GAME_EDUCATIONAL', 'ANDROID-GAME_MUSIC', 'ANDROID-GAME_PUZZLE',
              'ANDROID-GAME_RACING', 'ANDROID-GAME_ROLE_PLAYING','ANDROID-GAME_SIMULATION', 
              'ANDROID-GAME_SPORTS','ANDROID-GAME_STRATEGY', 'ANDROID-GAME_TRIVIA', 
              'ANDROID-GAME_WORD','ANDROID-HEALTH_AND_FITNESS', 'ANDROID-HOUSE_AND_HOME',
              'ANDROID-LIBRARIES_AND_DEMO', 'ANDROID-LIFESTYLE','ANDROID-MAPS_AND_NAVIGATION', 
              'ANDROID-MEDICAL','ANDROID-MUSIC_AND_AUDIO', 'ANDROID-NEWS_AND_MAGAZINES',
              'ANDROID-PARENTING', 'ANDROID-PERSONALIZATION', 'ANDROID-PHOTOGRAPHY',
              'ANDROID-PRODUCTIVITY', 'ANDROID-SHOPPING', 'ANDROID-SOCIAL','ANDROID-SPORTS', 
              'ANDROID-TOOLS', 'ANDROID-TRAVEL_AND_LOCAL','ANDROID-VIDEO_PLAYERS', 
              'ANDROID-WEATHER', '0', '1', '2', '3', '4','5', '6', '7', '8', '9', '10', 
              '11', '12', '13', '14', '15', '16', '17','18', '19', '20', '21', '22', '23')].copy(deep=True)


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