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
datafile = "processed_training_set_Vu_DM-v5.csv"
train_df = pd.read_csv(f"./data/{datafile}", index_col=0)

X_train, y_train, X_val, y_val , groups_train, groups_val, test_data = split_train_data(train_df, testsize=0.2)

#%%


user_choise = user_choose_train_or_load()
if  user_choise == 1:
    print(f"Training model with the following features: {train_df.columns}")
    # training and defining model
    datenow = datetime.now().strftime("%m-%d-%Y-%H-%M")
    savedir = "models/xgb_rankerv4_"+ datenow+ ".json"
    DEVICE = 'gpu_hist' if gpu_is_available() else "auto"

    # docs can be found here https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRanker 
    model = xgb.XGBRanker(  
        tree_method=DEVICE,
        booster='gbtree',
        objective='rank:pairwise',
        random_state=42, 
        learning_rate=0.11,
        colsample_bytree=0.9, 
        eta=0.05, 
        max_depth=6, 
        n_estimators=300, 
        subsample=0.75,
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


#%%
#%%
# predicting on validation set

y_val_predict = model.predict(X_val)
X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]

X_val_res['predict'] = y_val_predict
X_val_res['scores'] = y_val['scores'].values

ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
val_ndcg = evaluate.evaluate_score(ndcg_df)
print("validation_ndcg: ", val_ndcg)
# evaluated_df.head()

# %%

test_df = pd.read_csv("data/processed_test_set_VU_DM-v3.csv")
# %%
df = test_df[["srch_id","prop_id"]]
df["predict"] = model.predict(test_df.loc[:, ~test_df.columns.isin(["srch_id"])].values)
# %%
# df.groupby("srch_id", sort=True)
# df.sort_values(by=["srch_id", "predict"], ascending=False)
sorted_df = df.sort_values(by=["srch_id", "predict"], ascending=[True,False])
# %%
# sorted_df[["srch_id", "prop_id"]].to_csv(f"predictions/VU-DM-2022-Group-155-pred-20-05-2-22-11-50.csv", index=False)
sorted_df[["srch_id", "prop_id"]].to_csv(f"predictions/VU-DM-2022-Group-155-pred-{datenow}.csv", index=False)
# %%
