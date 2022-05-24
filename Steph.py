#%%
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from torch.cuda import is_available as gpu_is_available

import evaluate
from utils import split_train_data, user_choose_model_to_load, user_choose_train_or_load


#%%
datafile = "processed_training_set_Vu_DM-v6.csv"
train_df = pd.read_csv(f"./data/{datafile}", index_col=0, nrows=3E5)

X_train, y_train, X_val, y_val , groups_train, groups_val, test_data = split_train_data(train_df, testsize=0.2)

#%%

DEVICE = 'gpu_hist' if gpu_is_available() else "auto"

model = xgb.XGBRanker(  
        tree_method=DEVICE,
        booster='gbtree',
        objective='rank:pairwise',
        random_state=42, 
        learning_rate=0.11,
        colsample_bytree=0.9, 
        eta=0.032, 
        gamma=2,
        max_depth=5, 
        n_estimators=300, 
        subsample=0.8,
        )


model.fit(X_train, y_train, group=groups_train, verbose=True)


#%%
import matplotlib.pyplot as plt

sorted_idx = model.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance", fontsize = 12)
plt.yticks(fontsize=7)


#%%

import shap
shap.initjs()
X_test_sampled = X_val.sample(200, random_state=10)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sampled)

#%%
shap.summary_plot(shap_values, X_test_sampled, plot_type="bar")
shap.summary_plot(shap_values, X_test_sampled)

#%%
y_val_predict = model.predict(X_val)


#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_val_predict)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])


#%%
print("attempt 2")

#%%

import lightgbm as lgb

DEVICE = 'gpu_hist' if gpu_is_available() else "auto"

gbm1 = lgb.LGBMRanker(
        tree_method=DEVICE,
        max_bin = 300,
        booster='gbtree',
        objective='lambdarank',
        random_state=42, 
        learning_rate=0.11,
        colsample_bytree=0.9, 
        eta=0.032, 
        gamma=2,
        max_depth=5, 
        n_estimators=300, 
        subsample=0.8,
        )


#%%

gbm1.fit(X_train, y_train, group=groups_train, verbose=True)

#%% 

y_val_predict = gbm1.predict(X_val)
X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]

X_val_res['predict'] = y_val_predict
X_val_res['scores'] = y_val['scores'].values

ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
val_ndcg = evaluate.evaluate_score(ndcg_df)
print("validation_ndcg: ", val_ndcg)

#%%
shap.initjs()
X_test_sampled = X_val.sample(200, random_state=10)

explainer = shap.TreeExplainer(gbm1)
shap_values = explainer.shap_values(X_test_sampled)

#%%
shap.summary_plot(shap_values, X_test_sampled, plot_type="bar")
shap.summary_plot(shap_values, X_test_sampled)


#%%
sorted_idx = gbm1.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], gbm1.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance", fontsize = 12)
plt.yticks(fontsize=7)

#%%
gbm1._best_score



#print('Training set score: {:.4f}'.format(gbm1.score(X_train, y_train)))
#%%
