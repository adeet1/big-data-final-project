#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
import dask.bag as db
import tqdm 

from distributed import Client
#from dask_jobqueue import SLURMCluster
from IPython.display import display

import os
from glob import glob



# Import the ratings dataset
ratings_df = pd.read_csv('ratings.csv').sort_values(by="timestamp").reset_index(drop=True)
#print("This dataset contains {} rows and {} columns.".format(ratings_df.shape[0], ratings_df.shape[1]))
#ratings_df.head()



# Count the number of ratings for each movie
num_ratings_per_movie = ratings_df.groupby("movieId").count()["rating"]

# Only keep movies with at least 5 ratings (e.g. we don't want to treat a movie that was rated 5 stars by too few people to be considered popular)
ratings_df = ratings_df.join(num_ratings_per_movie, on = "movieId", rsuffix="_count")
ratings_df = ratings_df[ratings_df["rating_count"] >= 5]
ratings_df.drop(columns=["rating_count"], inplace=True)

# Compute how many times each movie has been rated
#print(ratings_df["movieId"].value_counts(ascending=True))


# ## Training-validation-test split

# Split each user's data into training-validation-test set. First, we will separate 20% of the users to be part of the test set. Then for each remaining user, use 80% of their ratings for training and 20% for validation. We do this so that our recommender system generalizes across different kinds of users.


train_users = np.random.choice(ratings_df['userId'].unique(), int(len(ratings_df['userId'].unique()) * 0.8), replace=False)

train_val_df = ratings_df[ratings_df['userId'].isin(train_users)]
test_df = ratings_df[~ratings_df['userId'].isin(train_users)]



# For each user, compute the number of ratings they submitted
num_ratings_per_user = train_val_df.groupby('userId').count()['rating']


# Create training and validation sets for each user
train_df, val_df = pd.DataFrame(), pd.DataFrame()

for userId, num_ratings in tqdm.tqdm_notebook(list(zip(num_ratings_per_user.index, num_ratings_per_user))):
    # Get all the ratings for this user
    user_ratings = train_val_df[train_val_df['userId'] == userId].reset_index(drop=True)
    
    # Make the first 80% of this user's ratings the training set
    index_train = int(0.8*num_ratings)
    user_train = user_ratings.loc[:index_train-1, :]
    
    # Make the other 20% of this user's ratings the validation set
    user_val = user_ratings.loc[index_train:, :]
    
    # Add this user's individual training and validation sets to the unified training and
    # validation sets, respectively
    train_df = pd.concat([train_df, user_train], axis=0)
    val_df = pd.concat([val_df, user_val], axis=0)


train_df.to_csv("train_df_full.csv")
val_df.to_csv("val_df_full.csv")
test_df.to_csv("test_df_full.csv")

