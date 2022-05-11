#%%
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt

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