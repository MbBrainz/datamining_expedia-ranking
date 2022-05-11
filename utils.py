from pandas import DataFrame, to_datetime


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
        val += 5
    if row["click_bool"] == 1: 
        val += 1
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
            return df

    df.date_time = to_datetime(df.date_time)
    
    # df['year']  = df.date_time.dt.year
    df['month'] = df.date_time.dt.month 
    df['day']   = df.date_time.dt.day
    df['hour']  = df.date_time.dt.hour
    # df['minute']= df.date_time.dt.minute
    
    df.drop(columns="date_time", inplace=True)
    
    return df