#%%
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 

# %%
train_df = pd.read_csv("./data/training_set_VU_DM.csv")
test_df = pd.read_csv("./data/test_set_VU_DM.csv")

raw_df = pd.DataFrame(pd.concat([train_df, test_df]))
# %%
raw_df.head()
raw_df.info()
# %%
na_df = raw_df.isnull().sum()

drop_columns = na_df[na_df.values > 4E5].index

#%%
df = raw_df.drop(columns=drop_columns)
df.head()
print(df.columns)

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
# See how many booked places have a significant difference int the price they're booked for compared to what price they show.
# 
# if the gross booking value is not significantly different then price*staylength,
# then the gross_booking price can be
# estimated as that and then the gross_usd_booking doesnt add information
# This may allow us to eliminate the gross_booking_usd 


# find correlation between booking float and usd  value
booked_data = train_df[(train_df["booking_bool"] == True) & (train_df["price_usd"] > 1)]
#%%

booked_data["rel_usd_diff_per_night"] = (booked_data["gross_bookings_usd"] / booked_data["srch_length_of_stay"] - booked_data["price_usd"]) / booked_data["price_usd"]
booked_data["usd_diff"] = booked_data["gross_bookings_usd"] - booked_data["price_usd"]

booked_price_data = booked_data[["srch_length_of_stay" ,"gross_bookings_usd", "price_usd", "usd_diff", "rel_usd_diff_per_night"]]

#%%
# sns.histplot(booked_data, x="price_usd", )
sns.displot(booked_data, x="price_usd", )

# %%
# booked_price_data.sort_values(by="rel_usd_diff_per_night", axis='columns')
booked_price_data.sort_values("rel_usd_diff_per_night", ascending=False)

# %%
