#%%
from cmath import log
from distutils.log import info
from email.utils import collapse_rfc2231_value
from pickle import FALSE
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sympy import comp
from torch import isin

from utils import drop_features_with_many_na, get_features_from_datetime 

# %%
train_df = pd.read_csv("./data/training_set_VU_DM.csv")
test_df = pd.read_csv("./data/test_set_VU_DM.csv")

#%%
train_df[train_df["srch_id"] < 500].to_csv("./data/small_test_set_VU_DM.csv")

raw_df = pd.DataFrame(pd.concat([train_df, test_df]))
# %%
raw_df.head()
raw_df.info()
# %%
display(raw_df.isnull().sum())
df = drop_features_with_many_na(raw_df, 4E6)
df.head()
print(df.columns)

#%%
train_df = get_features_from_datetime(train_df)
  
#%%
# going to be investigating the COMP_ data and see what we can do with it

compdf = train_df.iloc[:,26:50]
compdf.head()
#%%
#Determining the humber of Null values 

competitordf = pd.read_csv("./data/small_test_set_VU_DM.csv")
#%%
competitordf.head()
compdf = competitordf.iloc[:,28:52]

compdf.head()
#%%

null_vals = compdf.isnull().sum()

null_vals.tolist()

#%%
cols = compdf.columns.tolist()
cols
#%%
plt.style.use("seaborn")
plt.plot(cols, null_vals)
plt.xticks(rotation=90)
#%%
compdf


#%%
ax = sns.boxplot(data=compdf, showfliers = True )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
#%%
ax = sns.boxplot(data=compdf, showfliers = False )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
#%%

ax = sns.boxenplot(data=compdf)
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
#%%

ax = sns.catplot(data=compdf)
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()

#%%
#finding the maximum values and potential outliers
#Columns for percent_diff need to be removed 


col1 = "comp1_rate_percent_diff"
col2 = "comp2_rate_percent_diff"
col3 = "comp3_rate_percent_diff"
col4 = "comp4_rate_percent_diff"
col5 = "comp5_rate_percent_diff"
col6 = "comp6_rate_percent_diff"
col7 = "comp7_rate_percent_diff"
col8 = "comp8_rate_percent_diff"

#%%
cols_replace = [col1, col2, col3, col4, col5, col6, col7, col8]
cols_replace
#%%
#finding the maximum values of the potential outliers
#Columns for huge variations will be replaced by the median value 
maxlist = compdf.max()
maxlist["comp1_rate_percent_diff"]
#%%

#finding the location of the outliers and investigating them further
max_vals_location = compdf.idxmax(axis = 0)
max_vals_location["comp1_rate_percent_diff"]

#%%
info = pd.DataFrame(data = compdf.describe())
info

#%%
info.loc["25%",col1]

#%%
#creating copies for the various scenarios 

compdf_lower_q = compdf.copy()
compdf_lower_q

#%%
#Filling all NaN values with the first quartile 
#seeing how it affects the plots and the data 
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
#The second option would be to lower the extreme values and replace these with the upper quartile
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

for column in  compdf_lower_q1:
    # print(column)
    # print(compdf_lower_q1.at[max_vals_location[column],column])
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


# ax = sns.boxplot(data=compdf_lower_q, showfliers = False )
# sns.despine(offset=10, trim=True)
# labels = ax.get_xticklabels()
# plt.setp(labels, rotation=85)
# plt.tight_layout()

#%%
compdf_lower_q


#%%
ax = sns.boxenplot(data=compdf_lower_q, showfliers = False )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=85)
plt.tight_layout()
















#--------------------------------------------------------------------------------






#%%
#%%
corr_df = df.drop(columns=["srch_id"])
corr_df.columns = [str(x).replace("_", " ").title() for x in corr_df.columns]
corr = corr_df.corr()
#%%
#%%
plt.figure(figsize=(16,6))
mask = np.triu(np.ones_like(corr))
heatmap = sns.heatmap(corr.round(3),cmap="BrBG", annot=True, mask=mask)
# plt.savefig("figures/correlation_hm_plot.pdf", bbox_inches="tight")
#%%
non_numericals = [x for x in df.columns if ( x.endswith("bool") | x.endswith("id"))]
# non_numericals = non_numericals + ["src_id",]
# %%
pair_df = df.drop(columns=non_numericals).drop(columns=drop_columns, errors='ignore')
# %%
# sns.pairplot(pair_df, hue="prop_starrating")
pair_df.columns
# %%
user_country_df = raw_df
user_country_df["count"] = 1
user_country_df = user_country_df[["prop_country_id", "visitor_location_country_id", "count"]].groupby(by=["prop_country_id", "visitor_location_country_id"]).count()

user_country_pivot_df = user_country_df.reset_index([0,1]).pivot("prop_country_id", "visitor_location_country_id", "count")
# %%

x = "prop_country_id"
y = "visitor_location_country_id"
sns.scatterplot(data=user_country_df, x=x, y=y)
# sns.histplot(data=user_country_df,x=x, y=y, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(data=user_country_df, x=x, y=y, levels=5, color="b", linewidths=4)
# %%
# See how many booked places have a significant difference in the price they're booked for compared to what price they show.
# 
# if the gross booking value is not significantly different then price*staylength,
# then the gross_booking price can be
# estimated as that and then the gross_usd_booking doesnt add information
# This may allow us to eliminate the gross_booking_usd 


# find correlation between booking float and usd  value
booked_data = train_df[(train_df["booking_bool"] == True) & (train_df["price_usd"] > 1) & (train_df["price_usd"] < 1E7)]
#%%

booked_data["rel_usd_diff_per_night"] = (booked_data["gross_bookings_usd"] / booked_data["srch_length_of_stay"] - booked_data["price_usd"]) / booked_data["price_usd"]
booked_data["usd_diff"] = booked_data["gross_bookings_usd"] - booked_data["price_usd"]
# booked_data["person_count"]

booked_price_data = booked_data[['srch_adults_count', 'srch_children_count',"srch_length_of_stay" ,"gross_bookings_usd", "price_usd", "usd_diff", "rel_usd_diff_per_night"]]

#%%
# sns.histplot(booked_data, x="price_usd", )
# sns.displot(booked_data, x="price_usd", )

# %%
# booked_price_data.sort_values(by="rel_usd_diff_per_night", axis='columns')
sorted_rel_usd_df = booked_price_data.sort_values("rel_usd_diff_per_night", ascending=False)

# %%
import tqdm

dis_df = df.drop(columns=non_numericals +["date_time"])
# fig, axes = plt.subplots(nrows=len(dis_df.columns), figsize=(30,15))
for col in tqdm.tqdm( dis_df.columns):
  sns.displot(dis_df[(dis_df["price_usd"] > 1) & (dis_df["price_usd"] < 1E7)][col], bins=20, kde=True)
  plt.tight_layout() 
  plt.show()
# plt.show()
# %%
sns.displot()
