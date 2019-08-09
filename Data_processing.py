# -*- coding: utf-8 -*-
from Function import*
import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


train_data = read_csv_file('./venv/Tencent/train.csv', logging=False)
ad = read_csv_file('./venv/Tencent/ad.csv', logging=False)

app_categories = read_csv_file('./venv/Tencent/app_categories.csv', logging=False)
app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(categories_process_first_class)
app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(categories_process_second_class)

# print(app_categories.head())
user = read_csv_file('./venv/Tencent/user.csv', logging=False)
user[user.age != 0].describe()
# user.age.value_counts()
# user
user = read_csv_file('./venv/Tencent/user.csv', logging=False)
user['age_process'] = user['age'].apply(process_province)
user["hometown_province"] = user['hometown'].apply(process_province)
user["hometown_city"] = user['hometown'].apply(process_city)
user["hometown_province"] = user['residence'].apply(process_province)
user["residence"] = user['residence'].apply(process_city)
# user.info()

# 合并数据
# train data
train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
train_data['clickTime_hour'] = train_data['clickTime'].apply(get_time_hour)

# test_data
test_data = read_csv_file('./venv/Tencent/test.csv')
test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)
test_data['clickTime_hour'] = test_data['clickTime'].apply(get_time_hour)

train_user = pd.merge(train_data, user, on='userID')
test_data = pd.merge(test_data, user, on='userID')
train_user_ad = pd.merge(train_user, ad, on='creativeID')
test_user_ad = pd.merge(test_data, ad, on='creativeID')
train_user_ad_app = pd.merge(train_user_ad, app_categories, on='appID')
test_user_ad_app = pd.merge(test_user_ad, app_categories, on='appID')
# print(train_user_ad_app.head())

# 取出数据和label
# 特征部分
x_user_ad_app = train_user_ad_app.loc[:, ['creativeID', 'userID', 'positionID', 'connectionType',
                                          'telecomsOperator', 'clickTime_day', 'clickTime_hour', 'age',
                                          'gender', 'education', 'marriageStatus', 'haveBaby',
                                          'residence', 'age_process', 'hometown_province', 'hometown_city',
                                          'residence_province', 'residence_city', 'adID', 'camgaignID', 'advertiserID',
                                          'appID', 'appPlatform', 'app_categories_first_class',
                                          'app_categories_second_class']]
x_test_clean = train_user_ad_app.loc[:, ['creativeID', 'userID', 'positionID', 'connectionType',
                                          'telecomsOperator', 'clickTime_day', 'clickTime_hour', 'age',
                                          'gender', 'education', 'marriageStatus', 'haveBaby',
                                          'residence', 'age_process', 'hometown_province', 'hometown_city',
                                          'residence_province', 'residence_city', 'adID', 'camgaignID', 'advertiserID',
                                          'appID', 'appPlatform', 'app_categories_first_class',
                                          'app_categories_second_class']].values
x_user_ad_app = x_user_ad_app.values
x_user_ad_app = np.array(x_user_ad_app, dtype='int32')
x_test_clean = np.array(x_test_clean, dtype='int32')
# 标签部分
y_user_ad_app = train_user_ad_app.loc[:, ['label']].values
#

