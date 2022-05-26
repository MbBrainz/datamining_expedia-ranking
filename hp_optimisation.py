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
VERSION = 13
datafile = f"processed_training_set_Vu_DM-v{VERSION}.csv"

train_df = pd.read_csv(f"./data/{datafile}", index_col=0, nrows=3E5)

import optuna
X_train, y_train, X_val, y_val , groups_train, groups_val, test_data= split_train_data(train_df, testsize=0.2)

def objective(trial: optuna.Trial):
    
    ### These parameters have been tuned, see v6 PDF's
    subsample = 0.8 # best performing
    subsample = trial.suggest_float("subsample", 0.6, 0.9)
    # max_depth = trial.suggest_int("max_depth", 3, 7)
    max_depth = 4 # 4 is consistantly the best 
    # learning_rate = trial.suggest_float('learning_rate', 0.1, 0.15)
    learning_rate = 0.115 # best performing 
    
    ### The optuna parameter search will be performed on the following h. parameters:
    eta = 0.032
    eta = trial.suggest_float("eta", 0.0, 0.05)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    n_estimators = trial.suggest_int('n_estimators', 250, 350)
    
    ### This parameter might still be promising
    # rank_objective = trial.suggest_categorical("objective", ["rank:pairwise", "rank:ndcg"])
    rank_objective = "rank:pairwise" # waaaaay better then ncdg
    
    model = XGBRanker(  
        # tree_method='gpu_hist',
        booster='gbtree',
        objective=rank_objective,
        random_state=7,
        gamma=gamma,
        learning_rate=learning_rate,
        colsample_bytree=0.9, 
        eta=eta, 
        max_depth=max_depth, 
        n_estimators=n_estimators, 
        subsample=subsample
        )

    model.fit(X_train, y_train, group=groups_train, verbose=True)
    
    y_val_predict = model.predict(X_val)
    X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]
    #%%

    X_val_res['predict'] = y_val_predict
    X_val_res['scores'] = y_val['scores'].values

    ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
    val_ndcg = evaluate_score(ndcg_df)
    trial.report(val_ndcg, 1)
    
    ### The objective value for the optuna stydy
    return val_ndcg
    
    
study = optuna.create_study(direction="maximize") 
timer_time = time()

#%%
"""Starts parameter search..."""
### number of trials is the amount of total points it will check. 
### Optuna starts randomly within the domain, but concentrates searches around spaces with more promising results.
study.optimize(objective, n_trials=100, show_progress_bar=True)   
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
optuna.visualization.plot_slice(study)
#%%