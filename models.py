import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import collect_list

spark = SparkSession.builder.appName('part1').getOrCreate()

# ==========================================================================
# IMPORT DATA
# ==========================================================================

train_df_name = "ratings-small-train.csv"
val_df_name = "ratings-small-val.csv"

train_df = pd.read_csv(train_df_name)
print("Train data:", train_df.shape)

val_df = pd.read_csv(val_df_name)
print("Val data:", val_df.shape)

# ==========================================================================
# BASELINE MODEL
# ==========================================================================

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
users.sort()
n_users = len(users)

# Number of movies that we recommend to each user
n_recs = 100

# We need to recommend to everyone in the validation set the 100 movies that we learned from the training set
R_i = np.array(mean_ratings.head(n_recs).index)
R = [R_i.tolist()] * n_users # we recommend the same movies to everyone

# We create a list of arrays, where D[i] = an array of validation data of the highest-rated movies by user i
val_df_group = val_df.sort_values("rating", ascending=False).groupby("userId")
D = list(map(lambda user: val_df_group.get_group(user)["movieId"].values.astype(float).tolist(), users))

# Evaluate model performance
pred_and_labels = spark.sparkContext.parallelize(list(zip(R, D)))
metrics = RankingMetrics(pred_and_labels)
print("Baseline ----------------")
print("Precision:", metrics.precisionAt(n_recs))
print("MAP:", metrics.meanAveragePrecision)

# ==========================================================================
# ALS MODEL
# ==========================================================================

train_df = spark.read.csv(train_df_name, header=True, schema="rowIndex INT, userId DOUBLE, movieId DOUBLE, rating FLOAT, timestamp LONG")
val_df = spark.read.csv(val_df_name, header=True, schema="rowIndex INT, userId DOUBLE, movieId DOUBLE, rating FLOAT, timestamp LONG")

# Fit the model
als = ALS(maxIter=5, regParam=0.02, rank=75, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train_df)

# Compute the movie recommendations for all users
R = model.recommendForAllUsers(n_recs).select("userId", "recommendations.movieId")

# Get the movies we recommended for the validation users
R_val = R.filter(R.userId.isin(users.tolist()))
R_val = R_val.sort("userId")

# Evaluate model on the validation users
R_val = R_val.select("movieId").collect()
R_val = list(map(lambda row: row.movieId, R_val))
R_val = list(map(lambda arr: np.array(arr).astype("double").tolist(), R_val)) # convert ints to floats

pred_and_labels = spark.createDataFrame(list(zip(R_val, D)), "prediction: array<double>, label: array<double>")

evaluator = RankingEvaluator()
print("ALS ----------------")
print("Precision:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "precisionAtK", evaluator.k: 100}))
print("MAP:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "meanAveragePrecision", evaluator.k: 100}))
