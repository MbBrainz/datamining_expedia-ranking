#%%
from time import time
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GroupKFold, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import make_scorer, ndcg_score, pairwise_distances
from xgboost import XGBRanker
from evaluate import evaluate_score

from utils import split_train_data, split_train_data_nsplits
#%%
train_df = pd.read_csv("./data/processed_training_set_Vu_DM-v3.csv", index_col=0, nrows=5E5)
#%%
# gkf = GroupKFold(n_splits=5)
# groups = train_df["srch_id"]
# x_train = train_df.loc[:, ~train_df.columns.isin(['scores'])]
# y_train = train_df.loc[:, train_df.columns.isin(['scores'])]

# cv = gkf.split(x_train, y_train, groups=groups)
# cv_group = gkf.split(x_train, groups=groups)
# def group_gen(flatted_group, cv):
#     for train, _ in cv:
#         yield np.unique(flatted_group.iloc[train], return_counts=True)[1]
# gen = group_gen(groups, cv_group)
 
# # %%
# ranker = XGBRanker(random_state=42)

# params = { 'max_depth': [3,6,10],
# 'learning_rate': [0.01, 0.05, 0.1],
# 'n_estimators': [50 ,100, 500, 1000],
# 'eta':[0.01, 0.03, 0.05],
# 'colsample_bytree': [0.3, 0.7]}

# scoring = make_scorer(pairwise_distances, greater_is_better=True)

# search = HalvingGridSearchCV(ranker, 
#                              param_grid=params,
#                              verbose=2,
#                              scoring=scoring,
#                              n_jobs=14)

# # this generates error. Try to fix using https://forum.numer.ai/t/learning-to-rank/454/17 
# search.fit(x_train, y_train, group=next(gen))
# print("Best parameters:", search.best_params_)
# # %%

import optuna
X_train, y_train, X_val, y_val , groups_train, groups_val, test_data= split_train_data(train_df, testsize=0.2)

def objective(trial: optuna.Trial):
    
    max_depth = trial.suggest_int("max_depth", 5, 10)
    eta = trial.suggest_float("eta", 0.01, 0.05)
    
    n_estimators = 300
    learning_rate = 0.11
    n_estimators = trial.suggest_int('n_estimators', 200, 400)
    # learning_rate = trial.suggest_float('learning_rate', 0.1, 0.0)
    
    model = XGBRanker(  
        # tree_method='gpu_hist',
        booster='gbtree',
        objective='rank:pairwise',
        random_state=42, 
        learning_rate=learning_rate,
        colsample_bytree=0.9, 
        eta=eta, 
        max_depth=max_depth, 
        n_estimators=n_estimators, 
        subsample=0.2
        )
    
   
    # X_train, y_train, X_val, y_val , groups_train, groups_val, test_data= split_train_data_nsplits(train_df, testsize=0.2, nsplits=20)


    model.fit(X_train, y_train, group=groups_train, verbose=True)
    
    y_val_predict = model.predict(X_val)
    X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]
    #%%

    X_val_res['predict'] = y_val_predict
    X_val_res['scores'] = y_val['scores'].values

    ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
    val_ndcg = evaluate_score(ndcg_df)
    trial.report(val_ndcg, 1)
    
    return val_ndcg
    
    
study = optuna.create_study(direction="maximize") 
timer_time = time()
study.optimize(objective, n_trials=100)   
timer_time = time() - timer_time
print(timer_time)

#%%
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#%%
optuna.visualization.plot_contour(study)

# %%
