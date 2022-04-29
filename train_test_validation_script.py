# ====================================================================================
# IMPORT LIBRARIES
# ====================================================================================

import numpy as np
np.random.seed(0)
import pandas as pd
import itertools

# ====================================================================================
# IMPORT DATA + DATA PROCESSING
# ====================================================================================

# Import the ratings dataset
ratings_df = pd.read_csv('ml-latest/ratings.csv')
ratings_df = ratings_df.sort_values(by="timestamp")

# Count the number of ratings for each movie
num_ratings_per_movie = ratings_df.groupby("movieId").count()["rating"]

# Only keep movies with at least 5 ratings (e.g. we don't want to treat a movie that was rated 5 stars by too few people to be considered popular)
ratings_df = ratings_df.join(num_ratings_per_movie, on = "movieId", rsuffix="_count")
ratings_df = ratings_df[ratings_df["rating_count"] >= 5]
ratings_df.drop(columns=["rating_count"], inplace=True)

# ====================================================================================
# TRAIN/VAL/TEST SPLIT
# ====================================================================================

# An array containing the IDs of all users
all_users = ratings_df['userId'].unique()

# Shuffle the array, because we want to randomly split the users into training/val/test sets
np.random.shuffle(all_users)

# Out of all the users we have, we put the first 60% of them in the training set 
train_size = int(0.6 * len(all_users))
train_users = all_users[:train_size]
train_df = ratings_df[ratings_df['userId'].isin(train_users)]

# Put the next 20% in the validation set
val_size = train_size + int(0.2 * len(all_users))
val_users = all_users[train_size:val_size]
val_df = ratings_df[ratings_df['userId'].isin(val_users)]

# Put the last 20% in the testing set
test_users = all_users[val_size:]
test_df = ratings_df[ratings_df['userId'].isin(test_users)]

print("Sanity check: {} ?= {}".format(len(ratings_df), len(train_df) + len(val_df) + len(test_df)))
print(len(train_df), len(val_df), len(test_df))

val_group = val_df.groupby('userId')
test_group = test_df.groupby('userId')

# A list of dataframes, where each dataframe has the ratings for one user
val_df_groups = list(map(lambda user: val_group.get_group(user), val_users))
test_df_groups = list(map(lambda user: test_group.get_group(user), test_users))

# For each validation user, get a chunk of their data
chunk_frac = 0.3
train_subsets = list(map(lambda user_data: user_data[:int(chunk_frac*len(user_data))], val_df_groups))

# Move these chunks into the training set
train_df = pd.concat([train_df] + train_subsets)

# Move these chunks out of the validation set
indices_to_drop = map(lambda subset: subset.index, train_subsets)
indices_to_drop = list(itertools.chain.from_iterable(indices_to_drop))
val_df = val_df.drop(indices_to_drop)

# For each testing user, get a chunk of their data
chunk_frac = 0.3
train_subsets = list(map(lambda user_data: user_data[:int(chunk_frac*len(user_data))], test_df_groups))

# Move these chunks into the training set
train_df = pd.concat([train_df] + train_subsets)

# Move these chunks out of the testing set
indices_to_drop = map(lambda subset: subset.index, train_subsets)
indices_to_drop = list(itertools.chain.from_iterable(indices_to_drop))
test_df = test_df.drop(indices_to_drop)

print("Sanity check: {} ?= {}".format(len(ratings_df), len(train_df) + len(val_df) + len(test_df)))
print(len(train_df), len(val_df), len(test_df))

# ====================================================================================
# EXPORT DATASETS
# ====================================================================================

train_df.to_csv("ratings-big-train.csv")
val_df.to_csv("ratings-big-val.csv")
test_df.to_csv("ratings-big-test.csv")