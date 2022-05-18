#%%
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import make_scorer, ndcg_score
from xgboost import XGBRanker

from utils import split_train_data

train_df = pd.read_csv("./data/processed_training_set_Vu_DM.csv", index_col=0)
#%%
X_train, y_train, X_val, y_val , groups_train, groups_val, test_data= split_train_data(train_df, testsize=0.03)


# %%
ranker = XGBRanker(random_state=42)

params = { 'max_depth': [3,6,10],
'learning_rate': [0.01, 0.05, 0.1],
'n_estimators': [50 ,100, 500, 1000],
'eta':[0.01, 0.03, 0.05],
'colsample_bytree': [0.3, 0.7]}

scoring = make_scorer(ndcg_score, greater_is_better=True)

search = HalvingGridSearchCV(ranker, 
                             param_grid=params,
                             verbose=2,
                             scoring=scoring,
                             n_jobs=14)

# this generates error. Try to fix using https://forum.numer.ai/t/learning-to-rank/454/17 
search.fit(X_train, y_train, groups=groups_train)
print("Best parameters:", search.best_params_)
# %%
