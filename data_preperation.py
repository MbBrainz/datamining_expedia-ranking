"""This file contains the code that prepares the data for training.

First we only focus on the training data. The test data will be treated accordingly

The model of choise is XGBoost so the data should be in the following format:
- Features need to be numerical

"""
#%%
from inspect import getfile
import pandas as pd

from utils import add_scores, drop_features_with_many_na, get_features_from_datetime
raw_train_df = pd.read_csv("./data/training_set_VU_DM.csv") # this set only contains first 500 srch_id's -> quicker runs for developement

#%%

train_df = drop_features_with_many_na(raw_train_df, 2E5)
train_df = get_features_from_datetime(train_df)

print("filtered features: \n")
print(train_df.columns)


train_df = add_scores(train_df)
train_df.head()
# %%
train_df = get_features_from_datetime(train_df)
train_df.head()

#%%
train_df.to_csv("./data/processed_training_set_Vu_DM.csv")
# %%

raw_test_df = pd.read_csv("./data/test_set_VU_DM.csv")
#%%
test_df = get_features_from_datetime(raw_test_df)

features = ['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_log_historical_price', 'position',
       'price_usd', 'promotion_flag', 'srch_destination_id',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'random_bool', 'month', 'day', 'hour']

test_df = test_df[features]
test_df.to_csv("./data/processed_test_set_VU_DM.csv")
# %%
