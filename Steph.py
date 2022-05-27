#%%
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from torch.cuda import is_available as gpu_is_available
import shap
import matplotlib.pyplot as plt

import evaluate
from utils import split_train_data, user_choose_model_to_load, user_choose_train_or_load


#%%
datafile = "processed_training_set_Vu_DM-v12.csv"
train_df = pd.read_csv(f"./data/{datafile}", index_col=0)

#%%
train_df
#%%

X_train, y_train, X_val, y_val , groups_train, groups_val, test_data = split_train_data(train_df, testsize=0.2)

#%%

X_val
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
X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]

X_val_res['predict'] = y_val_predict
X_val_res['scores'] = y_val['scores'].values

ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
val_ndcg = evaluate.evaluate_score(ndcg_df)
print("validation_ndcg: ", val_ndcg)


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
        max_bin = 100,
        boosting='dart',
        num_leaves = 30,
        objective='lambdarank',
        random_state=42, 
        learning_rate=0.05,
        colsample_bytree=0.9, 
        eta=0.032, 
        gamma=2,
        max_depth=7, 
        n_estimators=400, 
        subsample=0.8,
        )

#%%

gbm1.fit(X_train, y_train, group=groups_train, verbose=True)

#%% 

y_val_predict = gbm1.predict(X_val)
X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]

X_val_res['predict'] = y_val_predict
X_val_res['scores'] = y_val['scores'].values

#%%
y_val_predict


#%%
ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
val_ndcg = evaluate.evaluate_score(ndcg_df)
print("validation_ndcg: ", val_ndcg)

#%%
shap.initjs()
X_test_sampled = X_val.sample(300, random_state=10)

explainer = shap.TreeExplainer(gbm1)
shap_values = explainer.shap_values(X_test_sampled)

#%%
X_test_sampled.columns
#%%
shap.summary_plot(shap_values, X_test_sampled, plot_type="bar")
shap.summary_plot(shap_values, X_test_sampled)

#%%


#%%

sorted_idx = gbm1.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], gbm1.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance", fontsize = 12)
plt.yticks(fontsize=7)

#%%

train_again =train_df.copy()
col_to_remove = ["srch_destination_id","srch_length_of_stay", "Competitor_Available_count", "srch_children_count","Competitor_Cheaper_count", "prop_brand_bool", "price_hist_diff", "star_hist_diff", "srch_booking_window"]

for i in col_to_remove:
        del train_again[i]

#%%
trainer = train_again.copy()

X_train, y_train, X_val, y_val , groups_train, groups_val, test_data = split_train_data(trainer, testsize=0.2)
gbm1.fit(X_train, y_train, group=groups_train, verbose=True)
#print('Training set score: {:.4f}'.format(gbm1.score(X_train, y_train)))
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
X_test_sampled = X_val.sample(300, random_state=10)

explainer = shap.TreeExplainer(gbm1)
shap_values = explainer.shap_values(X_test_sampled)

#%%
X_test_sampled.columns
#%%
shap.summary_plot(shap_values, X_test_sampled, plot_type="bar")
shap.summary_plot(shap_values, X_test_sampled)
#%%