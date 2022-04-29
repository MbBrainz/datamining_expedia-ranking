Improvements XGBoost:
- use historical booking counts as features (like: number of bookings in same cluster for same search destination)
- train xgboost within a holdout set (like last 4 months) that is not used to create the counts
- expand the train set to 100 cluster-rows for each booking, and then reduce it by using only a subset from the rows where cluster!=bookedcluster (this reduces complexity to 2-class-classification, and also reduces the number of columns - while increasing rows)







kaggle competition 2016:

1st place 0.60218 by "idle_speculation"

xgboost "rank:pairwise"

--> further reading: https://www.kaggle.com/c/expedia-hotel-recommendations/discussion/21607

- map user cities and clusters to latitude and longitude using gradient descent
- build a factorization machine model for each cluster
- calculate historical click and book rates by a variety of factors
- build a modified "rank:pairwise" xgboost model on 1-3


2nd place 0.53218 by "beluga"
- Map city to globe and calculate lat-long
- create seasonality proxy for destinations
- Hotel cluster frequencies based on factors
- user preferences


Split data based on leakage hotel matches 1:2
Trained binary xgb models separately for each hotel clusters.
I used 8-20% of the negative samples in each binary classifiers to speed up training.
separate feature selection and paropt helped.


-------------------



Leakage

 confirm there is a leak in the data and it has to do with the orig_destination_distance attribute. Thanks to Arnaud who was the first to point this out. Good catch!

Our estimate is this directly affects approx. 1/3 rd of the holdout data.

The contest will continue without any changes. For clarity, we are confirming you can find hotel_clusters for the affected rows by matching rows from the train dataset based on the following columns: user_location_country, user_location_region, user_location_city, hotel_market and orig_destination_distance. However, this will not be 100% accurate because hotels can change cluster assignments (hotels popularity and price have seasonal characteristics).



-------
Paper Marty

papers 2013

1st place Zhang

Gradient Boosting Machines with/without EXP features 
- NaN to negative  
- ??

-----




https://dl.acm.org/doi/pdf/10.1145/2347736.2347755

https://www.youtube.com/watch?v=OtD8wVaFm6E

https://www.kaggle.com/competitions/expedia-hotel-recommendations/code

https://www.kaggle.com/code/wendykan/map-k-demo/notebook

https://www.kaggle.com/code/jiaofenx/expedia-hotel-recommendations

https://www.kaggle.com/c/expedia-hotel-recommendations/discussion/21607

https://www.dataquest.io/blog/digitalocean-docker-data-science/

