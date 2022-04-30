import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics

spark = SparkSession.builder.appName('part1').getOrCreate()

train_df = pd.read_csv("ratings-small-train.csv")
print("Train data:", train_df.shape)

val_df = pd.read_csv("ratings-small-val.csv")
print("Val data:", val_df.shape)

"""
Compute the mean rating of each movie by grouping by movieId, and aggregating by mean.
Note that we don't want to explicitly compute the utility matrix, because doing so will
take a very long time, and the resulting matrix will be very large and take up a lot of
memory.

We compute the 100 highest mean-rated movies from the training set. We will recommend
these 100 movies to every single user in the validation set.
"""
# Compute mean ratings
mean_ratings = train_df[["movieId", "rating"]].groupby("movieId").mean()["rating"].sort_values(ascending=False)

# Number of users in the validation data
users = val_df['userId'].unique()
n_users = len(users)

# Number of movies that we recommend to each user
n_recs = 100

# We need to recommend to everyone in the validation set the 100 movies that we learned from the training set
R_i = np.array(mean_ratings.head(n_recs).index)
R = [R_i.tolist()] * n_users # we recommend the same movies to everyone

# We create a list of arrays, where D[i] = an array of validation data of the highest-rated movies by user i
val_df_group = val_df.sort_values("rating", ascending=False).groupby("userId")
D = list(map(lambda user: val_df_group.get_group(user)["movieId"].values[0:100].tolist(), users))

# Evaluate model performance
pred_and_labels = spark.sparkContext.parallelize(list(zip(R, D)))
metrics = RankingMetrics(pred_and_labels)
print(metrics.precisionAt(100))
print(metrics.meanAveragePrecision)
