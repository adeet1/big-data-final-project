# ====================================================================================
# IMPORT LIBRARIES
# ====================================================================================

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import tqdm 
import getpass

import os
from glob import glob
import pyspark
from pyspark.sql import SparkSession

# Create the spark session object
spark = SparkSession.builder.appName('part1').getOrCreate()

# ====================================================================================
# IMPORT DATA + DATA PROCESSING
# ====================================================================================

# Import the ratings dataset
ratings_df = spark.read.csv('hdfs:/user/ajp756/ratings-small.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp LONG')
ratings_df = ratings_df.sort("timestamp")

# Count the number of ratings for each movie
num_ratings_per_movie = ratings_df.groupby("movieId").count()

# Only keep movies with at least 5 ratings (e.g. we don't want to treat a movie that was rated 5 stars by too few people to be considered popular)
ratings_df = ratings_df.join(num_ratings_per_movie, on = "movieId")
ratings_df = ratings_df.filter("count >= 5")
ratings_df = ratings_df.select(["movieId", "userId", "rating", "timestamp"]) # we only needed the count column to filter out movies with few ratings, we don't need it anymore
print(ratings_df.show())

# ====================================================================================
# TRAIN/VAL/TEST SPLIT
# ====================================================================================

# An array containing the IDs of all users
all_users = ratings_df.select('userId').distinct().collect()
all_users = np.array(all_users).ravel()
print("all_users =", all_users)

# Shuffle the array, because we want to randomly split the users into training/val/test sets
np.random.shuffle(all_users)

# Out of all the users we have, we put the first 60% of them in the training set 
train_size = int(0.6 * len(all_users))
train_users = all_users[train_size]
train_df = ratings_df[ratings_df.userId.isin(list(train_users))]

# Put the next 20% in the validation set
val_size = train_size + int(0.2 * len(all_users))
val_users = all_users[train_size:val_size]
val_df = ratings_df[ratings_df.userId.isin(list(val_users))]

# Put the last 20% in the testing set
test_users = all_users[val_size:]
test_df = ratings_df[ratings_df.userId.isin(list(test_users))]

print("Sanity check: {} ?= {}".format(len(ratings_df), len(train_df) + len(val_df) + len(test_df)))

val_chunk_frac = 0.3
val_group = val_df.groupby('userId')
for user in tqdm.tqdm_notebook(val_users):
    val_user_data = val_group.get_group(user)

    # Take a chunk of data
    train_subset = val_user_data.loc[:int(val_chunk_frac*len(val_user_data)), :]

    # Move it into the training set
    train_df = train_df.union(train_subset)

    # Move it out of the validation set
    val_df = val_df.filter(val_df.index != train_subset.index)

test_chunk_frac = 0.3
test_group = test_df.groupby('userId')
for user in tqdm.tqdm_notebook(test_users):
    test_user_data = test_group.get_group(user)

    # Take a chunk of data
    train_subset = test_user_data.loc[:int(test_chunk_frac*len(test_user_data)), :]

    # Move it into the training set
    train_df = train_df.union(train_subset)

    # Move it out of the testing set
    test_df = test_df.filter(test_df.index != train_subset.index)

"""
# Get a list of the users we didn't end up sampling
remain_users = np.where(~np.isin(all_users, train_users))

# Out of the remaining users, randomly sample 50% of them to put in the validation set
val_users = np.random.choice(remain_users, int(len(remain_users) * 0.5), replace=False)

# Whoever is left over will be in the testing set
#test_users = 

train_df = ratings_df.filter(ratings_df.userId.isin(train_users))
test_df = ratings_df.filter(~ratings_df.userId.isin(train_users))

# For each user, compute the number of ratings they submitted
num_ratings_per_user = train_val_df.groupby('userId').count()
print(num_ratings_per_user.show())

# Create training and validation sets for each user
# Select 60% of users, and put all their data in training
# For the next 20% of users, split each user's data into two subsets: put one subset in training and the other subset in validation
# For the next 20% of users, split each user's data into two subsets: put one subset in training and the other subset in testing

train_df = spark.createDataFrame(data = spark.sparkContext.emptyRDD(), schema='userId INT, movieId INT, rating FLOAT, timestamp LONG')
val_df = spark.createDataFrame(data = spark.sparkContext.emptyRDD(), schema='userId INT, movieId INT, rating FLOAT, timestamp LONG')

for userId, num_ratings in tqdm.tqdm(list(zip(num_ratings_per_user.index, num_ratings_per_user))):
    # Get all the ratings for this user
    user_ratings = train_val_df.filter(train_val_df.userId == userId).reset_index(drop=True)
    
    # Make the first 80% of this user's ratings the training set
    index_train = int(0.8*num_ratings)
    user_train = user_ratings.loc[:index_train-1, :]
    
    # Make the other 20% of this user's ratings the validation set
    user_val = user_ratings.loc[index_train:, :]
    
    # Add this user's individual training and validation sets to the unified training and
    # validation sets, respectively
    train_df = pyspark.pandas.concat([train_df, user_train], axis=0)
    val_df = pyspark.pandas.concat([val_df, user_val], axis=0)

# Sanity check (these should match)
print(len(ratings_df))
print(sum([len(train_df), len(val_df), len(test_df)]))

# Compute mean ratings
mean_ratings = train_df[["movieId", "rating"]].groupby("movieId").mean()["rating"].sort_values(ascending=False)

# We need to recommend to everyone in the validation set the 100 movies that we learned from the training set
R_i = np.array(mean_ratings.head(100).index)
print(R_i)

# Number of users in the validation data
users = val_df['userId'].unique()
n_users = len(users)

# Number of movies that we recommend to each user
n_recs = 100

# Initialize array of relevances
rel_D = pd.DataFrame(np.empty(shape=(n_users, n_recs)), index=users, columns=R_i)

val_df = val_df.sort_values("rating", ascending=False)
val_df_group = val_df.groupby("userId")
D = list(map(lambda user: val_df_group.get_group(user)["movieId"].values[0:100], users))

# For each user, compute relevance
# rel_D[u, m] = 1 if recommended movie m is relevant to user u and 0 otherwise
rel_D[:] = np.row_stack(list(map(lambda D_i: np.isin(R_i, D_i).astype(int), D)))

# Compute precision at k=100
k = 100
precision = np.mean(np.sum(rel_D, axis=1) / k)
print("Precision: {}%".format(np.round(precision*100, 5)))

# Compute MAP
j_plus_1 = np.arange(1, 101)
inner_sum = np.sum(rel_D / j_plus_1, axis=1)
N = np.fromiter(map(lambda D_i: D_i.size, D), dtype=int) # an array where N[i] = number of movies rated by user i
MAP = np.mean(1/N * inner_sum)
print("The MAP is: {}%".format(np.round(MAP*100, 5)))
"""
