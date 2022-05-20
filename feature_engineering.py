from pandas import DataFrame, to_datetime
import numpy as np
from tqdm import tqdm

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
    print(f"Dropping these high-nan colums: {na_columns}")
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


def comp_inv_and_cheaper_count(df: DataFrame):
  
  comp_columns = [x for x in df.columns if str(x).startswith("comp")     & str(x).endswith("rate")]
  comp_diff_columns = [x for x in df.columns if str(x).startswith("comp")& str(x).endswith("rate_percent_diff")]
  comp_inv_columns = [x for x in df.columns if str(x).startswith("comp") & str(x).endswith("inv")]

  # df["Comp_cheaper_and_in_150percent"] = 0
  df["Competitor_Available_count"] = 0
  df["Competitor_Cheaper_count"] = 0
  
  for i, comp_rate_col in tqdm(enumerate(comp_columns), desc="comp", total=len(comp_columns)):
    
    cheaper_mask = (df[comp_rate_col] == -1)
    percentage_mask = (df[comp_diff_columns[i]] < 150)
    
    df.loc[cheaper_mask, "Competitor_Cheaper_count"] += 1
    
    # # This feature results in exactly the same column as the  "Competitor_Cheaper_count" feature
    # cheaper_and_b150_mask = cheaper_mask & percentage_mask
    # df.loc[cheaper_and_b150_mask, "Comp_cheaper_and_in_150percent"] += 1
    
    inv_equal_zero_mask = (df[comp_inv_columns[i]] == 0)
    df.loc[inv_equal_zero_mask, "Competitor_Available_count"] += 1
    
  return df.drop(columns=comp_columns + comp_diff_columns + comp_inv_columns)
    