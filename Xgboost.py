# Xgboost调参
import xgboost as xgb
from Data_processing import*
from sklearn.model_selection import GridSearchCV
param_id = {
    'max_depth': [3, 4, 5, 7, 9],
    'n_estimators': [10, 50, 100, 400, 800],
    'learning_rate': [0.1, 0.2, 0.3],
    'gamma': [0, 0.2],
    'subsample': [0.8, 1]
}

xgb_model = xgb.XGBClassifier()
xgs = GridSearchCV(xgb_model, param_id, n_jobs=-1)
xgs.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app[0],))
print(xgs.best_score_)
print(xgs.best_params_)
xgs.predict_proba(x_test_clean)