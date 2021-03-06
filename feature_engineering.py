from pandas import DataFrame, to_datetime
import numpy as np
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def remove_empty_searches(df: DataFrame):
    """ An Empty search is consider specific search id that has zero rows(properties/hotels) that are clicked or booked"""
    
    #TODO

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
    
    #convert to cyclic values (each a sin and each a cosine)
    df = convert_time_cyclic(df,"hour", 24)
    df = convert_time_cyclic(df,"day", 31)
    df = convert_time_cyclic(df,"month", 12)
    # df['minute']= df.date_time.dt.minute
    return df.drop(columns="date_time")

def convert_time_cyclic(df: DataFrame, time_var='hour', max=24) -> DataFrame:
    df[f'{time_var}_sin'] = np.sin(df[f'{time_var}'] / (max-1) * 2 * np.pi)
    df[f'{time_var}_cos'] = np.cos(df[f'{time_var}'] / (max-1) * 2 * np.pi)
    return df.drop(columns=[time_var])


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

def convert_price_to_log(df: DataFrame, drop_original=True):
    df_ = df.copy()
    df_["log_price_usd"] = np.log(df_["price_usd"])
    df_["log_price_usd"][df_["log_price_usd"] < 0] = -1
    if drop_original:
        return df_.drop(columns="price_usd")
    else:
        return df_
    # TODO: LOG OF PRICE 

def create_star_difference(train_df:DataFrame):
  """Creates a feature from the difference between the visitor_hist_starring and the prop_starring. If the visitor_hist_staring is Nan, this difference is set to 0

  Args:
      train_df (DataFrame): _description_

  Returns:
      DataFrame: dataframe with the new 'star_hist_diff' feature and without the visitor_hist_staring
  """ 
  star_df = train_df[["visitor_hist_starrating", "prop_starrating"]]
  not_has_hist_mask = star_df["visitor_hist_starrating"].isnull()
  valid = star_df[~not_has_hist_mask]
  star_df["star_hist_diff"] = 0
  star_df.loc[~not_has_hist_mask, "star_hist_diff"] = valid["visitor_hist_starrating"] - valid["prop_starrating"]
  train_df["star_hist_diff"] = star_df["star_hist_diff"]
  return train_df.drop(columns=["visitor_hist_starrating"])

def create_price_difference(train_df:DataFrame):
  """Creates a feature from the difference between the visitor_price_adr_hist and the price_usd. If the visitor_price_adr_hist is Nan, this difference is set to 0

  Args:
      train_df (DataFrame): _description_

  Returns:
      DataFrame: dataframe with the new 'price_hist_diff' feature and without the visitor_price_adr_hist
  """ 
  price_df = train_df[["visitor_hist_adr_usd", "price_usd"]]
  not_has_hist_mask = price_df["visitor_hist_adr_usd"].isnull()
  valid = price_df[~not_has_hist_mask]
  price_df["price_hist_diff"] = 0
  price_df.loc[~not_has_hist_mask, "price_hist_diff"] = valid["visitor_hist_adr_usd"] - valid["price_usd"]
  train_df["price_hist_diff"] = price_df["price_hist_diff"]
  return train_df.drop(columns=["visitor_hist_adr_usd"])

def create_ranked_feature(df:DataFrame, variable:str, ascending_bool=False):
    """Creates a variable that represents the rank of the property relative to the search. 

    Args:
        df (DataFrame): dataframe that contains the variable as column
        variable (str): variable to compute the relative rank for 
        ascending_bool (bool, optional): ascending or not. Defaults to False.

    Returns:
        DataFrame: dataframe with the new feature
    """
    df = df.join(df.groupby('srch_id')[[variable]].rank(ascending=ascending_bool).astype(int).add_suffix('_rank'))
    return df
    
def add_prop_count(df: DataFrame):
    """counts the frequency of a specific property id and adds it to all the rows with the corresponding property ids"""
    
    prop_df = df[["prop_id", "srch_id"]].groupby(by=["prop_id"]).count().reset_index().rename(columns={"srch_id":"prop_count"})
    return df.join(prop_df.set_index("prop_id"), on="prop_id")

def prop_avg_score(df:DataFrame):
  #replce the 0 values with NaN for the sake of calculating the mean
  df.prop_location_score1.replace(0, np.nan, inplace = True)
  d = df.groupby('prop_id').prop_location_score1.agg('mean').reindex(df.prop_id).reset_index()
  df["prop_location_score1_avg"] = d.prop_location_score1
  #
  df.prop_location_score1.replace(np.nan, 0, inplace = True) 

  return df

def properties_clicked_df(df:DataFrame):
    clicked_df = df[(df['click_bool'] > 0)]
    return clicked_df

# TODO: do something with the 2nd prop rev score 
# TODO: check hist price of user and price of that property id

def add_prop_feature_mean(df:DataFrame, features_to_mean=["promotion_flag", "prop_location_score2", "srch_query_affinity_score"]):
    """adds means of features per prop id"""
    
    prop_promo_df = df[["prop_id"] + features_to_mean]\
        .groupby(by=["prop_id"])\
        .agg(["mean"]).reset_index()
    prop_promo_df.columns = ['_'.join(col) for col in prop_promo_df.columns]
    prop_promo_df.rename(columns={"prop_id_":"prop_id"}, inplace=True)

    return df.join(prop_promo_df.set_index("prop_id"), on="prop_id")


def get_estimated_position_random_bol(df_with_position, df, train = True):
    
    """
    df_with_position: dataframe with training set to create estimated position
    df: dataframe with either training or test set
    train: Bool to adapt for only training set
    
    """
    
    estimated_position_random_1 = df_with_position.loc[df_train["random_bool"] == 1]
    estimated_position_random_0 = df_with_position.loc[df_train["random_bool"] == 0]
    
    
    estimated_position_random_1 = estimated_position_random_1.groupby(["srch_destination_id", "prop_id"]).agg({"position": "mean"})

    estimated_position_random_1 = estimated_position_random_1.rename(index=str, columns={"position": "estimated_position"}).reset_index()

    estimated_position_random_1["srch_destination_id"] = (estimated_position_random_1["srch_destination_id"].astype(str).astype(int))

    estimated_position_random_1["prop_id"] = (estimated_position_random_1["prop_id"].astype(str).astype(int))

    # estimated_position_random_1["estimated_position"] = (1 / estimated_position_random_1["estimated_position"])

    estimated_position_random_1["random_bool"] = 1
    
    
    estimated_position_random_0 = estimated_position_random_0.groupby(["srch_destination_id", "prop_id"]).agg({"position": "mean"})

    estimated_position_random_0 = estimated_position_random_0.rename(index=str, columns={"position": "estimated_position"}).reset_index()

    estimated_position_random_0["srch_destination_id"] = (estimated_position_random_0["srch_destination_id"].astype(str).astype(int))

    estimated_position_random_0["prop_id"] = (estimated_position_random_0["prop_id"].astype(str).astype(int))

    # estimated_position_random_0["estimated_position"] = (1 / estimated_position_random_0["estimated_position"])

    estimated_position_random_0["random_bool"] = 0
    
    
    estimated_position = pd.concat([estimated_position_random_1, estimated_position_random_0])
    
    df = df.merge(estimated_position, how="left", on=["srch_destination_id", "prop_id", "random_bool"])
    
    
    if train:
        print("Correlation between 'position' and 'estimated_position' :", df["position"].corr(df["estimated_position"]))
        
        df = df.drop('position', axis=1)
    
    return df