from pandas import DataFrame, to_datetime
from sklearn.model_selection import GroupShuffleSplit


def drop_features_with_many_na(df: DataFrame, threshold=2E6):
    """drop all features with more then `threshold` na values

    Args:
        df (pd.DataFrame): the dataframe to drop the columns for
        threshold (int, optional): the theashold. Defaults to 2E5.

    Returns:
        pd.DataFrame: returns dataframe without the dropped features
    """
    na_df = df.isnull().sum()
    na_columns = na_df[na_df.values > threshold].index
    return df.drop(columns=na_columns)

def add_scores(df: DataFrame, drop_train_only_features=True):
    df["scores"] = df.apply (lambda row: add_scores_row(row), axis=1)
    
    if drop_train_only_features:
        df = df.drop(columns=["booking_bool","click_bool"]) #, "gross_bookings_usd"])
    
    return df

def add_scores_row(row):
    '''adds SCORES to dataframe'''
    val = 0
    if row["booking_bool"] == 1: 
        val = 5
    elif row["click_bool"] == 1: 
        val = 1
    return val

def get_features_from_datetime(df: DataFrame) -> DataFrame:
    """converts date_time column into month, day and hour column and returns dataframe

    Args:
        df (DataFrame): dataframe with 'date_time' column

    Returns:
        DataFrame: DataFrame with 'month', 'day' and 'hour' column
    """
    if {'month', 'day', 'hour'}.issubset(df.columns):
        print("columns 'month', 'day', 'hour' are already present ")
        if 'date_time' in df.columns: 
            return df.drop(columns=['date_time'])
        else: 
            return df.copy(deep=True)

    df.date_time = to_datetime(df.date_time)
    
    # df['year']  = df.date_time.dt.year
    df['month'] = df.date_time.dt.month 
    df['day']   = df.date_time.dt.day
    df['hour']  = df.date_time.dt.hour
    # df['minute']= df.date_time.dt.minute
    return df.drop(columns="date_time")
    


def Comp_inv_and_Cheaper_count(df1):

  df = df1.copy() 
  
  perc_diff_cols = []
  for i in range(1,9,1):
    perc_diff_cols.append("comp{}_rate_percent_diff".format(i))

  comp_rate = []

  for i in range(1,9,1):
        comp_rate.append("comp{}_rate".format(i))

  total = []
  for i, row in df.iterrows():
    counter = []
    for j in comp_rate:
      if df.at[i, j] == -1:
        for k in perc_diff_cols:
          if df.at[i, k] < 150:
            counter.append(1)

    total.append(np.sum(counter))

  df["Comp_cheaper_and_in_150percent"] = total

  for i in perc_diff_cols:
    df.drop(i, axis = 1, inplace = True)

  comp_inv = []
  for i in range(1,9,1):
      comp_inv.append("comp{}_inv".format(i))
    
  comp_invdf =  df[comp_inv]
  row_count_inv = []
  
  for i, row in comp_invdf.iterrows():
    row_vals = row.tolist()
    row_total = []
    for i in row_vals:
      if i == 0:
        row_total.append(1)
    row_count_inv.append(np.sum(row_total))

  df["Competitor_Available_count"] = row_count_inv

  for i in comp_inv:
    df.drop(i, axis = 1, inplace = True)
   
  comp_ratedf =  df[comp_rate]
  row_count_rate = []
  
  for i, row in comp_ratedf.iterrows():
    row_value = row.tolist()
    row_total = []
    for i in row_value:
      if i == -1:
        row_total.append(1)
    row_count_rate.append(np.sum(row_total))

  df["Competitor_Cheaper_count"] = row_count_rate

  for i in comp_rate:
    df.drop(i, axis = 1, inplace = True)

  return df

import numpy as np 
"""UNUSED 
This function will be used to evaluate the scores, but we dont think we use it in the preprocessing
"""
def discountedCumulativeGain(result):
    dcg = []
    for idx, val in enumerate(result): 
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)


def split_train_data(train_df: DataFrame, testsize=0):
    gss = GroupShuffleSplit(test_size=testsize, n_splits=1, random_state = 7).split(train_df, groups=train_df['srch_id'])

    X_train_inds, X_test_inds = next(gss)

    train_data=  train_df.iloc[X_train_inds]
    X_train = train_data.loc[:, ~train_data.columns.isin(['scores'])]
    y_train = train_data.loc[:, train_data.columns.isin(['scores'])]

    groups_train = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    test_data= train_df.iloc[X_test_inds]

#We need to keep the id for later predictions
    X_test = test_data.loc[:, ~test_data.columns.isin(["scores"])]
    y_test = test_data.loc[:, test_data.columns.isin(['scores'])]

    groups_val = test_data.groupby('scores').size().to_frame('size')['size'].to_numpy()
    return X_train, y_train, X_test, y_test ,groups_train, groups_val,test_data

def USD_history_diff(df):

  df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(value = 0)
  df["hist_USD_diff"] = df["price_usd"] - df["visitor_hist_adr_usd"]

  return df