import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
import tqdm
sns.set_style("darkgrid")

spark = SparkSession.builder.appName('part1').getOrCreate()

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

# ==========================================================================
# LIGHTFM VS. ALS
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
    R_val = R_val.select("movieId").collect()
    R_val = list(map(lambda row: row.movieId, R_val))
    R_val = list(map(lambda arr: np.array(arr).astype("double").tolist(), R_val)) # convert ints to floats

    pred_and_labels = spark.sparkContext.parallelize(list(zip(R_val, D)))
    return pred_and_labels

lightfm_precision_k_list, lightfm_runtime_list = [], []
als_precision_k_list, als_runtime_list = [], []

if data_size == "small":
    percent_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
elif data_size == "big":
    percent_train = [0.001, 0.005, 0.01, 0.05, 0.1]

for pct in tqdm.tqdm(percent_train):
    print('Model with {}% of training data'.format(100*pct))
    sample_train = train_df.sample(fraction=pct, withReplacement=False)#.reset_index()
 
    # ALS model
    start_t = time()
    als_model = ALS(rank=110, regParam=0.01, userCol="userId",
                    itemCol="movieId", ratingCol="rating").fit(sample_train)
    end_t = time()
    pred_and_labels = evaluate_ALS(als_model)
    
    n_recs = 100     # number of movies that we recommend to each user
    als_precision_k_list.append(RankingMetrics(pred_and_labels).precisionAt(n_recs))
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
sns.lineplot(ax=axes[0], x=percent_train, y=lightfm_precision_k_list, color='green', marker='o', label="LightFM")
sns.lineplot(ax=axes[0], x=percent_train, y=als_precision_k_list, color='black', marker='o', label="ALS")
axes[0].set_title('Precision at k vs. % training set')
axes[0].set_xlabel('% training set')
axes[0].set_ylabel('Precision at k')
sns.lineplot(ax=axes[1], x=percent_train, y=lightfm_runtime_list, color='green', marker='o', label="LightFM")
sns.lineplot(ax=axes[1], x=percent_train, y=als_runtime_list, color='black', marker='o', label="ALS")
axes[1].set_title('Runtime vs. % training set')
axes[1].set_xlabel('% training set')
axes[1].set_yscale("log")
axes[1].set_ylabel('Runtime')
plt.legend()
plt.show()

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
