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


user_choise = user_choose_train_or_load()
if  user_choise == 1:
    print(f"Training model with the following features: {train_df.columns}")
    # training and defining model
    datenow = datetime.now().strftime("%m-%d-%Y-%H-%M")
    savedir = "models/xgb_rankerv6_opt_"+ datenow+ ".json"
    DEVICE = 'gpu_hist' if gpu_is_available() else "auto"

    # docs can be found here https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRanker 
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
    # model.fit(X_train, y_train, group=groups_train, eval_set=(X_val,y_val),eval_group=groups_val, verbose=True)
    # print(model.evals_result())
    
    model.save_model(savedir)
    
elif user_choise == 2:
    print("\n")
    model_dir = user_choose_model_to_load()
    print("loading model...")
    
    model_xgb_2 = xgb.XGBRanker()
    model_xgb_2.load_model(model_dir)
    model = model_xgb_2
    
elif user_choise == 3:
    print("quitting...")
    raise Exception()


#%%
import matplotlib.pyplot as plt

sorted_idx = model.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance", fontsize = 12)
plt.yticks(fontsize=7)

#%%
print("attempt 2")

#%%
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
import shap
shap.initjs()
X_test_sampled = X_val.sample(200, random_state=10)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sampled)

#%%
shap.summary_plot(shap_values, X_test_sampled, plot_type="bar")
shap.summary_plot(shap_values, X_test_sampled)

#%%
shap.dependence_plot("LSTAT", shap_values, X_test_sampled)

#%%
y_val_predict = model.predict(X_val)
X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]

X_val_res['predict'] = y_val_predict
X_val_res['scores'] = y_val['scores'].values

ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
val_ndcg = evaluate.evaluate_score(ndcg_df)
print("validation_ndcg: ", val_ndcg)

#%%

import matplotlib.pyplot as plt

sorted_idx = model.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance", fontsize = 12)
plt.yticks(fontsize=7)

#%%

import lightgbm as lgb

DEVICE = 'gpu_hist' if gpu_is_available() else "auto"

gbm1 = lgb.LGBMRanker(
        tree_method=DEVICE,
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
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])


#%%
y_val_predict = gbm1.predict(X_val)

model_lgb.fit(X_train.values, y_train)
pred = model_lgb.predict(X_val.values)
#%%
import shap
X_sampled = X_train.sample(50, random_state=10)
#%%
shap.initjs()
explainer = shap.TreeExplainer(model_lgb)
#%%
shap_values = explainer.shap_values(X_sampled)

#%%
shap.force_plot(explainer.expected_value, shap_values[0,:], X_sampled.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values, X_train)