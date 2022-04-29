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
sns.scatterplot(data=user_country_df, x="prop_country_id", y="visitor_location_country_id")

# %%
