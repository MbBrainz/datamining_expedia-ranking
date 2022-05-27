"""This file contains the code that prepares the data for training.

First we only focus on the training data. The test data will be treated accordingly

The model of choise is XGBoost so the data should be in the following format:
- Features need to be numerical

"""
#%%
import pandas as pd
from time import time
from utils import add_scores
from feature_engineering import add_prop_count, add_prop_feature_mean, convert_price_to_log, create_ranked_feature, drop_features_with_many_na, get_features_from_datetime, comp_inv_and_cheaper_count, create_price_difference, create_star_difference
import warnings

warnings.filterwarnings("ignore")

raw_train_df = pd.read_csv("./data/training_set_VU_DM.csv") # this set only contains first 500 srch_id's -> quicker runs for developement

#%%
VERSION = 13
datafile = f"processed_training_set_Vu_DM-v{VERSION}.csv"
testfile = f"processed_test_set_VU_DM-v{VERSION}.csv"

description = f""" version {VERSION} \n
added  
    "srch_room_count",
    "srch_children_count",
    "srch_adults_count",
    "srch_length_of_stay",
    
not removing nan values for prop review and location_score2
"""

data = [{ "version": VERSION, "description":description, "datafile":datafile, "testfile": testfile}]
pd.DataFrame.from_records(data).to_csv("data/datalog.csv", mode="a", index=False, header=False)
#%%

# iets met prop_id in entrire dataset 
# add prop_loc vs visitor loc bool
# booking during usa holyday bool
# booking during europe holyday bool
# rooms per adult 
# rooms per person
# try checking if the booking day is a weekend or not


train_df = raw_train_df

def add_features(data_df: pd.DataFrame):
    initial_features = data_df.columns
    data_df = add_prop_count(data_df)
    data_df = add_prop_feature_mean(data_df, 
                                    features_to_mean=[
                                        "promotion_flag",
                                        "prop_location_score2", 
                                        "srch_query_affinity_score",
                                        "srch_room_count",
                                        "srch_children_count",
                                        "srch_adults_count",
                                        "srch_length_of_stay",
                                        ])
    
    # storing the prop review and location score before  creating ranged 
    prop_review_score = data_df["prop_review_score"].values
    prop_location_score2 = data_df["prop_location_score2"].values
    data_df["prop_review_score"] = data_df["prop_review_score"].fillna(0)
    data_df["prop_location_score2"] = data_df["prop_location_score2"].fillna(0)
    
    data_df = create_ranked_feature(data_df, "price_usd")
    data_df = create_ranked_feature(data_df, "prop_starrating")
    data_df = create_ranked_feature(data_df, "prop_review_score")
    data_df = create_ranked_feature(data_df, "prop_location_score1")
    data_df = create_ranked_feature(data_df, "prop_location_score2")
    
    # setting original nan values back
    data_df["prop_review_score"] = prop_review_score 
    data_df["prop_location_score2"] = prop_location_score2 

    data_df = comp_inv_and_cheaper_count(data_df)

    data_df = create_price_difference(data_df)

    data_df = create_star_difference(data_df)
    
    data_df = convert_price_to_log(data_df)

    data_df = get_features_from_datetime(data_df)
    
    #this one is for version 14 based on test_features.py results that can be found in .feature_optimisation/
    # data_df.drop(columns=["srch_destination_id"], inplace=True)
    
    added_features =  set(data_df.columns) -  set(initial_features)
    
    print(f"\toriginal features: \n {initial_features}\n\n \tfeatures added: \n {added_features}\n")
    return data_df

train_df = add_features(train_df)

#%%
# train_df = drop_features_with_many_na(train_df, 2E5)
# getting all the features we will use in test and trainset
train_df = add_scores(train_df)
train_df.drop(columns=["position", "gross_bookings_usd"], inplace=True)
features = train_df.columns.to_list()


features.remove("scores")

# train_df.head()
train_df.describe(percentiles=[])
train_df.to_csv(f"./data/{datafile}")
# %%

test_df = pd.read_csv("./data/test_set_VU_DM.csv")

# test_df = drop_features_with_many_na(test_df, 2E5)
test_df = add_features(test_df)
#%%
test_df = test_df[features]
test_df.to_csv(f"./data/{testfile}")
test_df.head()
# %%


