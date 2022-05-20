from pandas import DataFrame, to_datetime
import numpy as np


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


def Comp_inv_and_Cheaper_count(df1: DataFrame):

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