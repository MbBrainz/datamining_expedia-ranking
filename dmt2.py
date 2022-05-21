#%%
from cmath import log
from distutils.log import info
from email.utils import collapse_rfc2231_value
from itertools import groupby
from operator import index
from pickle import FALSE
from tkinter import commondialog
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sympy import comp
from torch import isin

from utils import drop_features_with_many_na, get_features_from_datetime, Comp_inv_and_Cheaper_count 
#%%
#loadinmg the smaller set to test with 

def Load_csv(Name):

  name_to_load = Name
  df = pd.read_csv("./data/{}".format(Name))
  return df

#%%
competitordf  = Load_csv("small_test_set_VU_DM.csv")

#%%
competitordf.columns
#%%

def USD_history_add(df):

  df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(value = 0)
  df["hist_USD_diff"] = df["price_usd"] - df["visitor_hist_adr_usd"]

  return df

#%%

newdf = USD_history_add(competitordf)
newdf

#%%



book_and_propid = ["prop_id", "booking_bool", "click_bool"]

idf = newdf[book_and_propid]

#%%

click_and_book_count = []
click_but_not_booked = []
for i, row in idf.iterrows():
  if idf.at[i, "booking_bool"] == 1 and idf.at[i, "click_bool"] == 1:
    click_and_book_count.append(1)
  else:
    click_and_book_count.append(0)
  
  if idf.at[i, "click_bool"] == 1 and idf.at[i, "booking_bool"] == 0:
    print("clicked but not booked")
    click_but_not_booked.append(1)
  else:
    click_but_not_booked.append(0)
#%%

newdf["clicked_and_booked"] = click_and_book_count

#%%
sns.violinplot(data = newdf, x = "prop_id", y = "hist_USD_diff")


#%%
cols = newdf.columns.tolist()
newcols = []
for i in cols:
  if i == "Unnamed: 0":
    print("skip")
  else:
    newcols.append(i)

newdf = newdf[newcols]

for i, row in newdf.iterrows():
  if newdf.at[i, "hist_USD_diff"] < 0 and newdf.at[i, "clicked_and_booked"] == 1:
    print("yes this occurs at {}".format(i))
    print(newdf.at[i, "hist_USD_diff"])

#%%

count = []

for i, row in newdf.iterrows():
  if newdf.at[i, "hist_USD_diff"] < 0:
    count.append(1)

np.sum(count)
#%%
#%%
for i, row in idf.iterrows():
  if idf.at[i, "booking_bool"] == 1 and idf.at[i, "click_bool"] == 1:

#%%
#categorizing the data to only the COMP_ features 
competitordf.head()
compdf = competitordf.iloc[:,28:52]

compdf.head()

#%%

newdf =  Comp_inv_and_Cheaper_count(competitordf)

#%%
newdf
#%%
#Determining the humber of Null values 
null_vals = compdf.isnull().sum()

null_vals.tolist()
null_vals

#%%
cols = compdf.columns.tolist()
cols

#%%
null_vals1 = pd.DataFrame(compdf.isnull().sum()).reset_index()
null_vals1["values"] = null_vals1[0]
null_vals1.drop( columns = 0, inplace= True)
sns.barplot(data = null_vals1, x = "index", y = "values")
sns.despine(offset=10, trim=True)
plt.title("number of NaN values present for features", fontsize = 16)
plt.xlabel("Feature", fontsize = 14)
plt.ylabel("Number of NaN values oresent in dataset", fontsize = 14)
plt.xticks(rotation=85)

#%%
#plotting as a percentage
null_vals1["Percentage"] = null_vals1["values"]/6973
null_sorted = null_vals1.sort_values('Percentage')

#%%
sns.barplot(data = null_sorted, x = "Percentage", y = "index", palette=("Blues_d"))
sns.despine(offset=10, trim=True)
plt.title("number of NaN values present for features", fontsize = 16)
plt.xlabel("Percentage of NaN values to total columns", fontsize = 14)
plt.ylabel("Feature", fontsize = 14)
plt.xticks(rotation=85)


#%%
#from the above plot we can see that comp_inv columns have the most data whilst
#the comp_percentage_diff has the most NaN values
#+1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and 
# competitor 1 have availability; null signifies there is no competitive data
compdf.describe()

#%%
#----------------------------------------------------------------
print("new Section regarding aggregation")
#----------------------------------------------------------------------
#%%
#creating the names to isolate comp_inv

# def Comp_inv_and_Cheaper_count(df1):

#   df = df1.copy() 
  
#   perc_diff_cols = []
#   for i in range(1,9,1):
#     perc_diff_cols.append("comp{}_rate_percent_diff".format(i))

#   total = []
#   for i, row in df.iterrows():
#     counter = []
#     for j in comp_rate:
#       if df.at[i, j] == -1:
#         for k in perc_diff_cols:
#           if df.at[i, k] < 150:
#             counter.append(1)

#     total.append(np.sum(counter))

#   df["Comp_cheaper_and_in 150percent"] = total

#   for i in perc_diff_cols:
#     df.drop(i, axis = 1, inplace = True)

#   comp_inv = []
#   for i in range(1,9,1):
#       comp_inv.append("comp{}_inv".format(i))
    
#   comp_invdf =  df[comp_inv]
#   row_count_inv = []
  
#   for i, row in comp_invdf.iterrows():
#     row_vals = row.tolist()
#     row_total = []
#     for i in row_vals:
#       if i == 0:
#         row_total.append(1)
#     row_count_inv.append(np.sum(row_total))

#   df["Competitor_Available_count"] = row_count_inv

#   for i in comp_inv:
#     df.drop(i, axis = 1, inplace = True)
   
#   comp_ratedf =  df[comp_rate]
#   row_count_rate = []
  
#   for i, row in comp_ratedf.iterrows():
#     row_value = row.tolist()
#     row_total = []
#     for i in row_value:
#       if i == -1:
#         row_total.append(1)
#     row_count_rate.append(np.sum(row_total))

#   df["Competitor_Cheaper_count"] = row_count_rate

#   for i in comp_rate:
#     df.drop(i, axis = 1, inplace = True)

#   return df

#%%
#identifying outliers and extreme values which may influence training and model accuracy 

ax = sns.boxplot(data=compdf, showfliers = True )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.xlabel("Feature variable", fontsize = 14)
plt.ylabel("Variable values in dataset", fontsize = 14)
plt.title("Graphic representation of feature values per column", fontsize = 14)
plt.tight_layout()
#%%
ax = sns.boxplot(data=compdf, showfliers = False )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.xlabel("Feature variable", fontsize = 14)
plt.ylabel("Spread values in dataset", fontsize = 14)
plt.title("Boxplot of feature variables", fontsize = 14)
plt.tight_layout()
#%%
ax = sns.boxenplot(data=compdf)
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
#%%
#Before elimination we will look at possible correlations 
#looking at correlations greater than 0.8
corr = compdf.corr()
kot = corr[corr>=.8]
plt.figure(figsize=(12,8))
sns.heatmap(kot)
#%%
#Looking at the number of negative values in the dataset
#by definition only comp_rate can be negative
#investigating the neg values
compdf.lt(0).sum()
#%%
#looking only at the DF excluding the comp{}_rate features
to_el = []
for i in range(1,9,1):
    to_el.append("comp{}_rate".format(i))

negative_vals = compdf.copy()
for i in to_el:

  negative_vals.drop(i, axis=1, inplace=True)

negative_vals.lt(0).sum()
#%%
#firstly will drop the columns which have negative values
#no real way to justify changing the numbers or keeping them 
#can only be 1 or 0 
negative_vals
#%%
#iterating over each column and identifying which row has these negative values

rows_to_delete = []
for (colname,colval) in negative_vals.iteritems():
    for  i in range(0,6973,1):
      if negative_vals.at[i, colname] < 0:
        print("A negative value was found in row #{} and column {}".format(i, colname))
        rows_to_delete.append(i)
print("total rows to potentially remove = {} (with duplicates)".format(len(rows_to_delete)))
#%%
#removing duplicates from the list 
res_to_remove = []
[res_to_remove.append(x) for x in rows_to_delete if x not in res_to_remove]
print("total number of rows to remove = {}".format(len(res_to_remove)))

#%%
#removing these rows from the dataset
new_df = negative_vals.copy()

for i in res_to_remove:
  new_df.drop([i], axis = 0, inplace=True)

new_df
#%%
#finding the maximum values and potential outliers
#Columns for percent_diff need to be removed 

cols_replace = []
for i in range(1,9,1):
    cols_replace.append("comp{}_rate_percent_diff".format(i))

cols_replace
#%%
#finding the maximum values of the potential outliers
#Columns for huge variations will be replaced by the median value 
maxlist = new_df.max()
maxlist["comp1_rate_percent_diff"]
#%%

#finding the location of the outliers and investigating them further
max_vals_location = new_df.idxmax(axis = 0)
max_vals_location["comp1_rate_percent_diff"]

#%%
info = pd.DataFrame(data = compdf.describe())
info

#%%
#creating copies for the various scenarios 
compdf_lower_q = compdf.copy()
compdf_lower_q

#%%
#Filling all NaN values with the first quartile 
#seeing how it affects the plots and the data 
#This was done by the previoud group who won.
for column in compdf_lower_q:
    compdf_lower_q[column] = compdf_lower_q[column].fillna(info.loc["25%",column])
    
#%%
#testing if all NaN values have been replaced 
NUlls = compdf_lower_q.isnull().sum()
NUlls
#%%
#All Nans have been replaced
print("ALl NaN have been replcaed with the lower quartile")
  

#%%
#Seeing how this has effected the data
ax = sns.boxenplot(data=compdf_lower_q)
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
#%%
#Another option would be to mitigate the extreme values
#only concerned with the closest 5 competitiors - therefore concentrate on closer percent diff rates
#finding the maximum values of the potential outliers
#Columns for huge variations will be replaced by the median value 
maxlist =  compdf_lower_q.max()
maxlist
#%%
#finding the location of the outliers and investigating them further
max_vals_location = compdf_lower_q.idxmax(axis = 0)

max_vals_location
#%%
#New describing DF
newinfo = pd.DataFrame(data = compdf_lower_q.describe())
newinfo
#%%
compdf_lower_q1 = compdf_lower_q.copy()
compdf_lower_q1.describe()

#%%
#Just doing something generic, will confirm with you guys how we want to handle these values
#can either completely remove them or adjust them
for column in  compdf_lower_q1:
    if compdf_lower_q1.at[max_vals_location[column],column] > 600:
      print("The column is = {} and the value is {}".format(column,compdf_lower_q1.at[max_vals_location[column],column]))
      print(compdf_lower_q1.loc[max_vals_location[column],column]/3)
      compdf_lower_q1.at[max_vals_location[column],column] = compdf_lower_q1.loc[max_vals_location[column],column]/3
      print("The new value is = to {}".format(compdf_lower_q1.loc[max_vals_location[column],column]))


#%%
ax = sns.boxenplot(data=compdf_lower_q1, showfliers = False )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()

#%%
#can also take the log value of the columns to see how it influences spread
#now to test the log of the larger columns
logcol1 = "log_comp1_rate_percent_diff"
logcol2 = "log_comp2_rate_percent_diff"
logcol3 = "log_comp3_rate_percent_diff"
logcol4 = "log_comp4_rate_percent_diff"
logcol5 = "log_comp5_rate_percent_diff"
logcol6 = "log_comp6_rate_percent_diff"
logcol7 = "log_comp7_rate_percent_diff"
logcol8 = "log_comp8_rate_percent_diff"

logcols = [logcol1, logcol2, logcol3, logcol4, logcol5, logcol6, logcol7, logcol8]
#%%

log_compdf = compdf_lower_q1.copy()
log_compdf[logcols] = np.log(compdf_lower_q1[cols_replace]+1)

#%%
log_compdf
#%%
for i in cols_replace:
  print(i)
  log_compdf.drop([i], inplace = True,  axis = 1)

#%%
cols_list = log_compdf.columns.tolist()
refined_cols = cols_list[1:16]

log_compdf.min()
#from the output we can see that the minimum is -1 therefore aggregate the data 

log_compdf[refined_cols] = log_compdf[refined_cols] + 1
log_compdf.head()

#%%
final_log = np.log(log_compdf[refined_cols]+1)
final_log[logcols] = np.log(compdf_lower_q1[cols_replace]+1)

#%%
ax = sns.boxenplot(data=final_log[logcols], showfliers = False )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()

#%%
for column in compdf:
    if (column in cols_replace) == True:
      print(column)
      # compdf_lower_q.at[column] = np.log(compdf_lower_q[column] +1 )


#%%
compdf_lower_q


#%%
ax = sns.boxenplot(data=compdf_lower_q, showfliers = False )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
#--------------------------------------------------------------------------------
# %%
