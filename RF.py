from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from Data_processing import*
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# # 随机森林建模&&特征重要度排序
# # % matlabplotlib inline
# # import matplotlib.pyplot as plt
# # import lightgbm as lgb
# # print("plot feature importance...")
# # ax = lgb.plot_importance(gbm, max_min_features=10)
# # plt.show()
# # 用RF 计算特征重要度

feat_labels = np.array(['creativeID', 'userID', 'positionID', 'connectionType',
                        'telecomsOperator', 'clickTime_day', 'clickTime_hour', 'age',
                        'gender', 'education', 'marriageStatus', 'haveBaby',
                        'residence', 'age_process', 'hometown_province', 'hometown_city',
                        'residence_province', 'residence_city', 'adID', 'camgaignID', 'advertiserID',
                        'appID', 'appPlatform', 'app_categories_first_class',
                        'app_categories_second_class'])
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(x_user_ad_app, y_user_ad_app)
forest.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0],))
joblib.dump(forest, 'forest1.pkl')
pickle.dump(forest.pkl)
forest = joblib.load('forest1.pkl')
importances = forest.feature_importances_
print(importances)
indices = np.argsort(importances[::-1])
print(indices)

for f in range(x_user_ad_app.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
plt.title('Feature importance')
plt.bar(range(x_user_ad_app.shape[1]),
        importances[indices], color='lightblue',
        align='center')
plt.xticks(range(x_user_ad_app.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_user_ad_app.shape[1]])
plt.tight_layout()
plt.show()


# 随机森林RF调参
param_id = {
    'n_estimators': [10, 100, 500, 1000],
    'max_features': [0.6, 0.7, 0.8, 0.9]
}

rf = RandomForestClassifier()
rfc = GridSearchCV(rf, param_id, scoring='neg_log_loss', cv=3, n_jobs=2)
rfc.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app[0],))
print(rfc.best_score_)
print(rfc.best_params_)


