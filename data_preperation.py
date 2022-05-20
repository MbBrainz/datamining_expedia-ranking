"""This file contains the code that prepares the data for training.

First we only focus on the training data. The test data will be treated accordingly

The model of choise is XGBoost so the data should be in the following format:
- Features need to be numerical

"""
#%%
import pandas as pd
from utils import add_scores
from feature_engineering import drop_features_with_many_na, get_features_from_datetime

raw_train_df = pd.read_csv("./data/training_set_VU_DM.csv") # this set only contains first 500 srch_id's -> quicker runs for developement

#%%

train_df = drop_features_with_many_na(raw_train_df, 2E5)
train_df = get_features_from_datetime(train_df)

print("filtered features: \n")
print(train_df.columns)


train_df = add_scores(train_df)
train_df.drop(columns=["position"], inplace=True)
features = train_df.columns.to_list()
features.remove("scores")
train_df.head()

#%%
train_df.to_csv("./data/processed_training_set_Vu_DM.csv")
# %%

raw_test_df = pd.read_csv("./data/test_set_VU_DM.csv")
#%%
test_df = get_features_from_datetime(raw_test_df)

#%%
test_df = test_df[features]
test_df.to_csv("./data/processed_test_set_VU_DM.csv")
test_df.head()
# %%


