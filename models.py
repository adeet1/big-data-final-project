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
from sklearn.model_selection import ParameterGrid
import tqdm

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
als = ALS(maxIter=5, regParam=0.02, rank=75, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(train_df)

def evaluate_ALS(model, users=users, n_recs=100):
    
    # Compute the movie recommendations for all users
    R = model.recommendForAllUsers(n_recs).select("userId", "recommendations.movieId")

    # Get the movies we recommended for the validation users
    R_val = R.filter(R.userId.isin(users.tolist())).sort("userId")

    # Evaluate model on the validation users
    R_val = R_val.select("movieId").collect()
    R_val = list(map(lambda row: row.movieId, R_val))
    R_val = list(map(lambda arr: np.array(arr).astype("double").tolist(), R_val)) # convert ints to floats

    pred_and_labels = spark.createDataFrame(list(zip(R_val, D)), "prediction: array<double>, label: array<double>")

    return pred_and_labels

pred_and_labels = evaluate_ALS(model)

evaluator = RankingEvaluator()
print("ALS ----------------")
print("Precision:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "precisionAtK", evaluator.k: n_recs}))
print("MAP:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "meanAveragePrecision", evaluator.k: n_recs}))
print("NDCG:", evaluator.evaluate(pred_and_labels, {evaluator.metricName: "ndcgAtK", evaluator.k: n_recs}))

"""
# ==========================================================================
# HYPERPARAMETER TUNING (ALS)
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

# ==========================================================================
# EXTENSION 1: LIGHTFM MODEL
# ==========================================================================

def lightfm_preprocessing(train_df, val_df):
    train_df = train_df.toPandas()
    val_df = val_df.toPandas()
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    
    lfm_object = Dataset()
    lfm_object.fit(users=full_df['userId'].unique(), items=full_df['movieId'].unique())
    
    user_movie_pair = lambda df, i: (df["userId"][i], df["movieId"][i], df["rating"][i])
    
    data = pd.Series(np.arange(train_df.shape[0])).map(lambda i: user_movie_pair(train_df, i))
    train_interactions, train_w = lfm_object.build_interactions(data)
    
    data = pd.Series(np.arange(val_df.shape[0])).map(lambda i: user_movie_pair(val_df, i))
    val_interactions, val_w = lfm_object.build_interactions(data)
    
    return train_interactions, train_w, val_interactions, val_w
    
train_interactions, train_w, val_interactions, val_w = lightfm_preprocessing(train_df, val_df)
  
model = LightFM(loss='warp', no_components=75, user_alpha=0.02).fit(interactions=train_interactions, sample_weight=train_w, epochs=1)

precision_k = precision_at_k(model, val_interactions, k=100).mean()



"""
# ==========================================================================
# HYPERPARAMETER TUNING (LIGHTFM)
# ==========================================================================


#"{'item_alpha': 1e-08, 'learning_rate': 0.5, 'learning_schedule': 'adadelta', 'loss': 'bpr', 'max_sampled': 5, 'no_components': 32, 'user_alpha': 1e-12}"
# 0.17278689
parameters = {
    "no_components": [5, 32, 100],
    "learning_schedule": ["adagrad", "adadelta"],
    "loss": ["bpr", "warp"],
    "learning_rate": [0.01, 0.1, 0.5],
    "item_alpha": [1e-12, 1e-8],
    "user_alpha": [1e-12, 1e-8],
    "max_sampled": [5, 10],
    }

#"{'item_alpha': 1e-08, 'learning_rate': 0.01, 'learning_schedule': 'adadelta', 'loss': 'bpr', 'max_sampled': 5, 'no_components': 32, 'user_alpha': 1e-12}"
# 0.16631149
parameters = {
    "no_components": [5, 32, 100],
    "learning_schedule": ["adagrad", "adadelta"],
    "loss": ["bpr", "warp"],
    "learning_rate": [0.01, 0.1, 0.5],
    "item_alpha": [1e-8],
    "user_alpha": [1e-12],
    "max_sampled": [5, 10],
    }

#"{'item_alpha': 1e-08, 'learning_rate': 0.5, 'learning_schedule': 'adadelta', 'loss': 'bpr', 'max_sampled': 5, 'no_components': 25, 'user_alpha': 1e-12}"
#0.17344262
parameters = {
    "no_components": [25, 32, 40, 50, 75],
    "learning_schedule": ["adadelta"],
    "loss": ["bpr", "warp"],
    "learning_rate": [0.01, 0.1, 0.5],
    "item_alpha": [1e-8],
    "user_alpha": [1e-12],
    "max_sampled": [5, 10],
    }

#"{'item_alpha': 1e-08, 'learning_rate': 0.5, 'learning_schedule': 'adadelta', 'loss': 'bpr', 'max_sampled': 10, 'no_components': 20, 'user_alpha': 1e-12}"
#0.17032787
parameters = {
    "no_components": np.arange(15,30),
    "learning_schedule": ["adadelta"],
    "loss": ["bpr"],
    "learning_rate": [0.001, 0.01, 0.1, 0.25, 0.5, 1],
    "item_alpha": [1e-5, 1e-8, 1e-10],
    "user_alpha": [1e-10, 1e-12, 1e-14],
    "max_sampled": [5, 10],
    }

train_interactions, train_w, val_interactions, val_w = lightfm_preprocessing(train_df, val_df)
start_t = time()
gridsearch_results = {}
param_grid = ParameterGrid(parameters)
for dict_ in tqdm.tqdm(param_grid):
    model = LightFM(no_components=dict_['no_components'], learning_schedule=dict_['learning_schedule'], loss=dict_['loss'],
                    learning_rate=dict_['learning_rate'], item_alpha=dict_['item_alpha'], user_alpha=dict_['user_alpha'],
                    max_sampled=dict_['max_sampled'])
    
    model.fit(interactions=train_interactions, sample_weight=train_w, epochs=1)

    precision_k = precision_at_k(model, val_interactions, k=100).mean()
    gridsearch_results[str(dict_)] = precision_k
    print("{} -> Precision at k: {}".format(str(dict_), precision_k))
    
end_t = time()
print("The grid search took {} minutes".format(round((end_t-start_t)/60, 2)))
print("The best hyper-parameters are: {}, with an associated precision at k of {}".format(
    max(gridsearch_results, key=gridsearch_results.get), max(gridsearch_results.values())))
"""

# ==========================================================================
# LIGHTFM VS. ALS
# ==========================================================================

lightfm_precision_k_list, lightfm_runtime_list = [], []
als_precision_k_list, als_runtime_list = [], []

#percent_train = [0.001, 0.005, 0.01, 0.05, 0.1] # ENABLE THIS FOR THE BIG DATASET
percent_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
for pct in tqdm.tqdm(percent_train):
    print('Model with {}% of training data'.format(100*pct))
    sample_train = train_df.sample(fraction=pct, withReplacement=False)#.reset_index()
 
    # ALS model
    start_t = time()
    als_model = ALS(rank=110, regParam=0.01, userCol="userId",
                    itemCol="movieId", ratingCol="rating").fit(sample_train)
    end_t = time()
    pred_and_labels = evaluate_ALS(als_model)
    
    als_precision_k_list.append(evaluator.evaluate(pred_and_labels, {evaluator.metricName: "precisionAtK", evaluator.k: n_recs}))
    als_runtime_list.append(end_t - start_t)
    
    # LightFM model
    train_interactions, train_w, val_interactions, val_w = lightfm_preprocessing(sample_train, val_df)
        
    start_t = time()
    model = LightFM(no_components=32, loss='warp', user_alpha=0.02).fit(interactions=train_interactions, sample_weight=train_w, epochs=1)
    end_t = time()
    precision_k = precision_at_k(model, val_interactions, k=100).mean()
    
    print("Precision at k: {}%, runtime: {}s".format(round(100*precision_k, 2), round(end_t - start_t, 5)))
    
    lightfm_precision_k_list.append(precision_k)
    lightfm_runtime_list.append(end_t - start_t)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(ax=axes[0], x=percent_train, y=lightfm_precision_k_list, label="LightFM")
sns.lineplot(ax=axes[0], x=percent_train, y=als_precision_k_list, label="ALS")
axes[0].set_title('Precision at k vs. % training set')
axes[0].set_xlabel('% training set')
axes[0].set_ylabel('Precision at k')
sns.lineplot(ax=axes[1], x=percent_train, y=lightfm_runtime_list, label="LightFM")
sns.lineplot(ax=axes[1], x=percent_train, y=als_precision_k_list, label="ALS")
axes[1].set_title('Runtime vs. % training set')
axes[1].set_xlabel('% training set')
axes[1].set_ylabel('Runtime')
plt.legend()
plt.show()

"""
# ==========================================================================
# EXTENSION 2: COLD START PROBLEM
# ==========================================================================


tags_df = pd.read_csv('tags.csv')
movies_df = pd.read_csv('movies.csv', index_col='movieId')

movies_df['release_year'] = movies_df['title'].map(lambda x: x[-5:-1])
movies_df.drop('title', axis=1, inplace=True)

movies_df['genres'] = movies_df['genres'].map(lambda x: x.split('|'))
movies_df = movies_df.explode('genres')

tags_df.drop('timestamp', axis=1, inplace=True)
tags_df['tag'] = tags_df['tag'].map(lambda x: x.lower().split(' '))
tags_df = tags_df.explode('tag')
"""
pass