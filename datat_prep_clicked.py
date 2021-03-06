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

raw_train_df = pd.read_csv("./data/training_set_VU_DM.csv") # this set only contains first 500 srch_id's -> quicker runs for developement

#%%
VERSION = 12
datafile = f"processed_training_set_Vu_DM-v{VERSION}.csv"
testfile = f"processed_test_set_VU_DM-v{VERSION}.csv"

description = f""" version {VERSION} \n
added  
    "srch_room_count",
    "srch_children_count",
    "srch_adults_count",
    "srch_length_of_stay",
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
def downsample(df, k=1):
    """
    k determines how many negative points you want (click_bool = 0) for each positive point. Default is 1. 
    """
    df_majority = df[df.click_bool == 0]
    df_minority = df[df.click_bool == 1]
    
    sampled = df_majority.groupby("srch_id").sample(k) 
    
    df_downsampled = pd.concat([sampled, df_minority])
    df_downsampled.sort_values(by=['srch_id'], inplace=True)
    return df_downsampled


train_df = raw_train_df

clicked_df = downsample(train_df)

def add_features(data_df):
    
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

    data_df["prop_review_score"] = data_df["prop_review_score"].fillna(-1)
    data_df["prop_location_score2"] = data_df["prop_location_score2"].fillna(-1)
    data_df = create_ranked_feature(data_df, "price_usd")
    data_df = create_ranked_feature(data_df, "prop_starrating")
    data_df = create_ranked_feature(data_df, "prop_location_score1")
    data_df = create_ranked_feature(data_df, "prop_review_score")

    data_df = comp_inv_and_cheaper_count(data_df)

    data_df = create_price_difference(data_df)

    data_df = create_star_difference(data_df)

    
    data_df = convert_price_to_log(data_df)

    data_df = get_features_from_datetime(data_df)
    

    return data_df

train_df = add_features(clicked_df)


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


