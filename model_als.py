import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('part1').config("spark.driver.memory", "11g").getOrCreate()

# ==========================================================================
# IMPORT DATA
# ==========================================================================

data_size = "small"
hdfs = False

prefix = "hdfs:/user/ajp756/" if hdfs else ""

train_df = spark.read.csv(prefix + "ratings-" + data_size + "-train.csv", header=True,
                          schema="rowIndex INT, userId DOUBLE, movieId DOUBLE, rating FLOAT, timestamp LONG")
print("Train data imported")

val_df = spark.read.csv(prefix + "ratings-" + data_size + "-val.csv", header=True,
                        schema="rowIndex INT, userId DOUBLE, movieId DOUBLE, rating FLOAT, timestamp LONG")
print("Val data imported")

# ==========================================================================
# ALS MODEL
# ==========================================================================

# Number of users in the validation data
users = val_df.select("userId").distinct().collect()
users = np.array(users).ravel().astype(int)
users.sort()

# Number of movies that we recommend to each user
n_recs = 100

# We create a list of arrays, where D[i] = an array of validation data of the highest-rated movies by user i
val_df_group = val_df.sort("rating", ascending=False)
val_df_group = val_df_group.select("userId", "movieId")
D = val_df_group.groupby("userId").agg(F.collect_list("movieId").alias("movies_rated")).collect()
D = list(map(lambda row: row["movies_rated"], D))

def evaluate_ALS(model, users=users, n_recs=100):
    # Compute the movie recommendations for all users
    R = model.recommendForAllUsers(n_recs).select("userId", "recommendations.movieId")

    # Get the movies we recommended for the validation users
    R_val = R.filter(R.userId.isin(users.tolist())).sort("userId")

    # Evaluate model on the validation users
    R_val = R_val.select("movieId") 
    R_val = R_val.toPandas()
    R_val = R_val["movieId"].tolist()
    
    pred_and_labels = spark.sparkContext.parallelize(list(zip(R_val, D)))
    return pred_and_labels

model = None
if data_size == "small":
    als = ALS(maxIter=16, regParam=0.01, rank=110, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_df)
elif data_size == "big":
    # Load the fitted model from memory
    print("Loading model...")
    model = ALSModel.load("als_model_big/")
    print("Loading model... DONE")

pred_and_labels = evaluate_ALS(model)

metrics = RankingMetrics(pred_and_labels)
print("ALS --------------")
print("Precision:", metrics.precisionAt(n_recs))
print("MAP:", metrics.meanAveragePrecision)

# For validation users, compute squared loss for each user (how good our recommendations are)
val_pred = model.transform(val_df).select("userId", "movieId", "rating", "prediction")
val_pred = val_pred.toPandas()
val_pred.to_csv("val_pred.csv")

"""
# Fit the model (try different ranks)
rank_values = np.array([10, 20, 30, 40, 50, 75, 100, 125, 150])
precision_values = np.empty_like(rank_values).astype(float)
map_values = np.empty_like(rank_values).astype(float)
for i in range(rank_values.size):
    r = rank_values[i]
    als = ALS(maxIter=5, regParam=0.01, rank=r, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_df)

    pred_and_labels = evaluate_ALS(model)

    metrics = RankingMetrics(pred_and_labels)
    print("ALS rank {} ----------------".format(r))
    print("Precision:", metrics.precisionAt(n_recs))
    print("MAP:", metrics.meanAveragePrecision)

    precision_values[i] = metrics.precisionAt(n_recs)
    map_values[i] = metrics.meanAveragePrecision

print(rank_values)
print(precision_values)
print(map_values)

plt.figure()
plt.plot(rank_values, precision_values, label="Precision at k")
plt.plot(rank_values, map_values, label="MAP")
plt.show()

# Fit the model (try different regParams)
reg_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.])
precision_values = np.empty_like(reg_values).astype(float)
map_values = np.empty_like(reg_values).astype(float)
for i in range(reg_values.size):
    r = reg_values[i]
    als = ALS(maxIter=5, regParam=r, rank=110, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_df)

    pred_and_labels = evaluate_ALS(model)

    metrics = RankingMetrics(pred_and_labels)
    print("ALS regParam {} ----------------".format(r))
    print("Precision:", metrics.precisionAt(n_recs))
    print("MAP:", metrics.meanAveragePrecision)

    precision_values[i] = metrics.precisionAt(n_recs)
    map_values[i] = metrics.meanAveragePrecision

print(reg_values)
print(precision_values)
print(map_values)

plt.figure()
plt.plot(reg_values, precision_values, label="Precision at k")
plt.plot(reg_values, map_values, label="MAP")
plt.xscale("log")
plt.legend()
plt.show()
"""

"""
# ==========================================================================
# HYPERPARAMETER TUNING
# ==========================================================================
#"{'rank': 100, 'regParam': 0.01}"
#0.10893442
parameters = {
    "rank": np.arange(50,101,10),
    "regParam": [0.001, 0.01, 0.1]
    }
#{'rank': 110, 'regParam': 0.01},
#0.11245901
parameters = {
    "rank": np.arange(110,150,10),
    "regParam": [0.01]
    }
#{'rank': 110, 'regParam': 0.01},
#0.11245901
parameters = {
    "rank": np.arange(105,115,1),
    "regParam": [0.01]
    }
evaluator = RankingEvaluator()
start_t = time()
gridsearch_results = {}
param_grid = ParameterGrid(parameters)
for dict_ in tqdm.tqdm(param_grid):
    als_model = ALS(rank=dict_['rank'], regParam=dict_['regParam'],
                    userCol="userId", itemCol="movieId", ratingCol="rating").fit(train_df)
    
    pred_and_labels = evaluate_ALS(als_model)
    
    gridsearch_results[str(dict_)] = evaluator.evaluate(pred_and_labels, {evaluator.metricName: "precisionAtK", evaluator.k: n_recs})
    print("{} -> Precision at k: {}".format(str(dict_), evaluator.evaluate(pred_and_labels, {evaluator.metricName: "precisionAtK", evaluator.k: n_recs})))
    
end_t = time()
print("The grid search took {} minutes".format(round((end_t-start_t)/60, 2)))
print("The best hyper-parameters are: {}, with an associated precision at k of {}".format(
    max(gridsearch_results, key=gridsearch_results.get), max(gridsearch_results.values())))
"""
