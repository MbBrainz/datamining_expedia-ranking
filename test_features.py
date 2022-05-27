import os
from time import time
from tracemalloc import start
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

from evaluate import evaluate_score
from utils import split_train_data

def validate(X_val, y_val, model:xgb.XGBRanker):
    y_val_predict = model.predict(X_val)
    X_val_res = X_val.loc[:, ["srch_id", "prop_id"]]

    X_val_res['predict'] = y_val_predict
    X_val_res['scores'] = y_val['scores'].values

    ndcg_df = X_val_res[['srch_id', 'predict', 'scores']]
    val_ndcg = evaluate_score(ndcg_df)
    # print("validation_ndcg: ", val_ndcg)
    return val_ndcg

def create_model():
    model = xgb.XGBRanker(  
        booster='gbtree',
        objective='rank:pairwise',
        random_state=7, 
        learning_rate=0.125,
        colsample_bytree=0.8, 
        eta=0.03, 
        max_depth=4, 
        n_estimators=320,
        gamma=1.2,
        subsample=0.8
        )
    return model


added_features = ['srch_children_count_mean', 'log_price_usd', 'prop_location_score2_mean', 'day', 'Competitor_Available_count', 'price_usd_rank', 'star_hist_diff', 'prop_location_score1_rank', 'prop_review_score_rank', 'srch_room_count_mean', 'month', 'srch_query_affinity_score_mean', 'promotion_flag_mean', 'Competitor_Cheaper_count', 'prop_starrating_rank', 'hour', 'price_hist_diff', 'prop_count', 'srch_length_of_stay_mean', 'srch_adults_count_mean']
added_features = ['visitor_location_country_id', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'promotion_flag', 'srch_destination_id',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'srch_query_affinity_score', 'orig_destination_distance', 'random_bool',
       'prop_count', 'promotion_flag_mean', 'prop_location_score2_mean',
       'srch_query_affinity_score_mean', 'srch_room_count_mean',
       'srch_children_count_mean', 'srch_adults_count_mean',
       'srch_length_of_stay_mean', 'price_usd_rank', 'prop_starrating_rank',
       'prop_review_score_rank', 'prop_location_score1_rank',
       'prop_location_score2_rank', 'Competitor_Available_count',
       'Competitor_Cheaper_count', 'price_hist_diff', 'star_hist_diff',
       'log_price_usd', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
       'month_sin', 'month_cos']
# 'srch_destination_id',
print(f"computing benchmark score...")

starttime = time()
VERSION = 13
datafile = f"processed_training_set_Vu_DM-v{VERSION}.csv"
train_df = pd.read_csv(f"./data/{datafile}", index_col=0, nrows=7E5)
X_train, y_train, X_val, y_val , groups_train, groups_val, test_data = split_train_data(train_df, testsize=0.2)
model= create_model()
model.fit(X_train, y_train, group=groups_train, verbose=True)
benchmark_score = validate(X_val, y_val, model)
print(f"The model with all features has score: {benchmark_score}")
print(f"starting to try to remove the following features: {added_features} \n")

scores = []
scores.append({"score":benchmark_score, "column_deleted": []})
deleted_columns = []

for col in tqdm(added_features, "feature ", position=0):
    copy = train_df.drop(columns=[col] + deleted_columns)
    X_train, y_train, X_val, y_val , groups_train, groups_val, test_data = split_train_data(copy, testsize=0.2)

    model = create_model()
    model.fit(X_train, y_train, group=groups_train, verbose=True)
    
    score = validate(X_val, y_val, model)
    this_record={'score':score,'column deleted': [col] + deleted_columns}
    if score > benchmark_score:
        benchmark_score = score
        deleted_columns.append(col)
    scores.append(this_record)
    
endtime = time()
print(f"time elapsed: {endtime-starttime}")
print("all scores:")
for score in scores:
    print(f"{score}")        
    
os.makedirs('feature_optimisation', exist_ok=True)  
pd.DataFrame.from_records(scores).to_csv(f'feature_optimisation/feature_optimisation2.csv', index=False)
