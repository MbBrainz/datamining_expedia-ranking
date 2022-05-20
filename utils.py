from time import sleep
from pandas import DataFrame, to_datetime
from sklearn.model_selection import GroupShuffleSplit


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
    

def split_train_data(train_df: DataFrame, testsize=0):
    # random state is 7, please dont change this
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



import os


def user_choose_train_or_load() -> bool:

  while True:
    print("Choose an option:")
    print("1. Train model")
    print("2. Load model")
    print("3. quit")
    sleep(1)
    
    choice = input("Enter your choice: ")
    if choice == "1":
        return 1
        
    elif choice == "2":
        return 2
      
    elif choice == "3":
        return 3
    else:
        print("Invalid choice, try again")
        
def user_choose_model_to_load(models_dir="models/"):
    while True:
      print("Choose a file to load the model from:")
      
      files = os.listdir(models_dir)
      for i, file in enumerate(files):
          print(f"[{i}]: {file}")
          
      choice = int(input())
      try:
        file = files[choice]
        return str(models_dir + file)
      
      except IndexError:
          print("That's not a valid choice!")
          continue

