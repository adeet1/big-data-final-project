import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import collect_list

spark = SparkSession.builder.appName('part1').getOrCreate()

# ==========================================================================
# IMPORT DATA
# ==========================================================================

data_size = "small"

train_df = pd.read_csv("ratings-" + data_size + "-train.csv")
print("Train data:", train_df.shape)

val_df = pd.read_csv("ratings-" + data_size + "-val.csv")
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
print("NDCG:", metrics.ndcgAt(n_recs))
        

# ==========================================================================
# ALS MODEL
# ==========================================================================

train_df = spark.read.csv("ratings-" + data_size + "-train.csv", header=True, schema="rowIndex INT, userId DOUBLE, movieId DOUBLE, rating FLOAT, timestamp LONG")
val_df = spark.read.csv("ratings-" + data_size + "-val.csv", header=True, schema="rowIndex INT, userId DOUBLE, movieId DOUBLE, rating FLOAT, timestamp LONG")

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
print("Precision:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "precisionAtK", evaluator.k: n_recs}))
print("MAP:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "meanAveragePrecision", evaluator.k: n_recs}))
print("NDCG:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "ndcgAtK", evaluator.k: n_recs}))


# ==========================================================================
# LIGHTFM MODEL
# ==========================================================================

def lightfm_preprocessing(train_df, val_df):
    
    full_df = pd.concat([train_df, val_df])
    
    lfm_object = Dataset()
    lfm_object.fit(users=full_df['userId'].unique(), items=full_df['movieId'].unique())
    
    user_movie_pair = lambda df, i: (df["userId"][i], df["movieId"][i], df["rating"][i])
    
    data = pd.Series(np.arange(train_df.shape[0])).map(lambda i: user_movie_pair(train_df, i))
    train_interactions, train_w = lfm_object.build_interactions(data)
    
    data = pd.Series(np.arange(val_df.shape[0])).map(lambda i: user_movie_pair(val_df, i))
    val_interactions, val_w = lfm_object.build_interactions(data)
    
    return train_interactions, train_w, val_interactions, val_w
    

precision_k_list, runtime_list = [], []

#percent_train = [0.001, 0.005, 0.01, 0.05, 0.1] # ENABLE THIS FOR THE BIG DATASET
percent_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
for pct in percent_train:
    print('Model with {}% of training data'.format(100*pct))
    sample_train = train_df.sample(frac=pct, replace=False).reset_index()
    
    if sample_train.isnull().values.any():
        print('Check null values')
    
    train_interactions, train_w, val_interactions, val_w = lightfm_preprocessing(sample_train, val_df)
        
    start_t = time()
    model = LightFM(loss='warp', no_components=75, user_alpha=0.02).fit(interactions=train_interactions, sample_weight=train_w, epochs=1)
    end_t = time()
    precision_k = precision_at_k(model, val_interactions, k=100).mean()
    print("Precision at k: {}%, runtime: {}s".format(round(100*precision_k, 2), round(end_t - start_t, 5)))
    
    precision_k_list.append(precision_k)
    runtime_list.append(end_t - start_t)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(ax=axes[0], x=percent_train, y=precision_k_list)
axes[0].set_title('Precision at k vs. % training set')
sns.lineplot(ax=axes[1], x=percent_train, y=runtime_list)
axes[1].set_title('Runtime vs. % training set')
