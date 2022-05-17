import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot

data_size = "big"

ratings_df = pd.read_csv("ratings-" + data_size + "-train.csv")
print("Train data:", ratings_df.shape)

df1 = ratings_df.groupby('movieId')['rating'].sum() / 5
df2 = ratings_df.groupby('movieId')['timestamp'].count().rename('count')
popularity = pd.concat([df1, df2], axis=1).sort_index()

def formula(S, N):
    a = 1 + S
    b = 1 + N - S
    
    first = a / (a+b)
    denominator = (a + b)**2 * (a + b + 1)
    second = 1.65 * np.sqrt((a * b) / denominator)

    return first - second

popularity['true_value'] = popularity.apply(lambda x: formula(x['rating'], x['count']), axis=1)
popularity = pd.concat([popularity, ratings_df.groupby('movieId')['rating'].mean()], axis=1)

popularity = popularity.sort_values('true_value', ascending=False)

chunk = int(len(popularity) / 10)
class_ = np.concatenate([np.ones(chunk)*1, np.ones(chunk)*2, np.ones(chunk)*3, np.ones(chunk)*4, np.ones(chunk)*5,
                         np.ones(chunk)*6, np.ones(chunk)*7, np.ones(chunk)*8, np.ones(chunk)*9, np.ones(chunk+5)*10])

popularity['rank'] = class_

low_rank_index = popularity[popularity['rank'].isin([6, 7, 8, 9, 10])].index
popularity.drop(low_rank_index, inplace=True)

latent_factors = pd.read_csv('latent_factors.csv', index_col='id').sort_index()
latent_factors.drop(low_rank_index, inplace=True)

latent_features = latent_factors['features'].map(lambda x: np.fromstring(x[1:-1], sep=','))
latent_features = np.row_stack(latent_features)

mapper = umap.UMAP(n_neighbors=15, min_dist=1, spread=1, random_state=101).fit(latent_features)
umap.plot.points(mapper, labels=popularity['rank'], theme='fire')
