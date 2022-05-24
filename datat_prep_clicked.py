#%%
"""This file contains the code that prepares the data for training.

First we only focus on the training data. The test data will be treated accordingly

The model of choise is XGBoost so the data should be in the following format:
- Features need to be numerical

"""
#%%
import pandas as pd
from time import time
from utils import add_scores
from feature_engineering import properties_clicked_df,convert_price_to_log, create_ranked_feature, drop_features_with_many_na, get_features_from_datetime, comp_inv_and_cheaper_count, create_price_difference, create_star_difference

raw_train_df = pd.read_csv("./data/training_set_VU_DM.csv") # this set only contains first 500 srch_id's -> quicker runs for developement

#%%

train_df = raw_train_df

train_df["prop_review_score"] = train_df["prop_review_score"].fillna(0)
train_df["prop_location_score2"] = train_df["prop_location_score2"].fillna(0)

train_df = properties_clicked_df(train_df)

train_df = create_ranked_feature(train_df, "price_usd")
train_df = create_ranked_feature(train_df, "prop_starrating")
train_df = create_ranked_feature(train_df, "prop_location_score1")
train_df = create_ranked_feature(train_df, "prop_review_score")

train_df = comp_inv_and_cheaper_count(train_df)

train_df = create_price_difference(train_df)

train_df = create_star_difference(train_df)

train_df = convert_price_to_log(train_df)

train_df = get_features_from_datetime(train_df)


train_df = drop_features_with_many_na(train_df, 2E5)
# getting all the features we will use in test and trainset
train_df = add_scores(train_df)
train_df.drop(columns=["position"], inplace=True)
features = train_df.columns.to_list()


features.remove("scores") 

# train_df.head()
train_df.describe(percentiles=[])

#%%
train_df.to_csv("./data/processed_training_set_Vu_DM-v10.csv")
# %%

test_df = pd.read_csv("./data/test_set_VU_DM.csv")

#%%
# test_df = get_features_from_datetime(raw_test_df)
# test_df = comp_inv_and_cheaper_count(test_df)

test_df["prop_review_score"] = test_df["prop_review_score"].fillna(0)
test_df["prop_location_score2"] = test_df["prop_location_score2"].fillna(0) # TODO Try with -1

### Ranked variables: See function description
test_df = create_ranked_feature(test_df, "price_usd")
test_df = create_ranked_feature(test_df, "prop_starrating")
test_df = create_ranked_feature(test_df, "prop_location_score1")
test_df = create_ranked_feature(test_df, "prop_review_score")

test_df = comp_inv_and_cheaper_count(test_df)

test_df = create_price_difference(test_df)

test_df = create_star_difference(test_df)

test_df = convert_price_to_log(test_df)

test_df = get_features_from_datetime(test_df)


test_df = drop_features_with_many_na(test_df, 2E5)

#%%
test_df = test_df[features]
test_df.to_csv("./data/processed_test_set_VU_DM-v10.csv")
test_df.head()
# %%
