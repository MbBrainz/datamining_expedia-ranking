#%%
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit


def split_train_data(train_df: DataFrame):
    gss = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 7).split(train_df, groups=train_df['srch_id'])

    X_train_inds, X_test_inds = next(gss)

    train_data=  train_df.iloc[X_train_inds]
    X_train = train_data.loc[:, ~train_data.columns.isin(['srch_id','scores'])]
    y_train = train_data.loc[:, train_data.columns.isin(['scores'])]

    groups_train = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    test_data= train_df.iloc[X_test_inds]

#We need to keep the id for later predictions
    X_test = test_data.loc[:, ~test_data.columns.isin(['srch_id', "scores"])]
    y_test = test_data.loc[:, test_data.columns.isin(['scores'])]

    groups_val = test_data.groupby('scores').size().to_frame('size')['size'].to_numpy()
    return X_train,y_train,groups_train,test_data

#%%
train_df = pd.read_csv("./data/processed_training_set_Vu_DM.csv", index_col=0)

X_train, y_train, groups_train, test_data = split_train_data(train_df)

#%%

# docs can be found here https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRanker 
model = xgb.XGBRanker(  
    # tree_method='gpu_hist',
    booster='gbtree',
    objective='rank:pairwise',
    random_state=42, 
    learning_rate=0.1,
    colsample_bytree=0.9, 
    eta=0.05, 
    max_depth=6, 
    n_estimators=110, 
    subsample=0.2 
    )

model.fit(X_train, y_train, group=groups_train, verbose=True)

#%%
def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])
  
predictions = (test_data.groupby('srch_id')
               .apply(lambda x: predict(model, x)))
# test data isnt in the right format jet
# official testset needs to be processed still
# need to write/copy evauation function
test_data.groupby("srch_id").apply(lambda x: x)




# %%
