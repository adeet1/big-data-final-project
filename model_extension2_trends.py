import pandas as pd

data_size = "big"
hdfs = False

prefix = "hdfs:/user/ajp756/" if hdfs else ""

val_pred = pd.read_csv("val_pred.csv")

# Compute squared loss for each rating-prediction pair
val_pred["sq_loss"] = (val_pred["rating"] - val_pred["prediction"])**2

# Compute average squared loss for each user
sq_loss_per_user = val_pred[["userId", "sq_loss"]].groupby("userId").mean().squeeze()
sq_loss_per_user = sq_loss_per_user.sort_values(ascending=False)
sq_loss_per_user.index = sq_loss_per_user.index.astype(int)

# For the users with the N_worst worst squared losses, look at the movies they rated and
# the genres of those movies, as well as the number of movies rated by those users
N_worst = 50
worst_losses = sq_loss_per_user.head(N_worst)
worst_pred_users = sorted(worst_losses.index.to_list())
worst_pred = val_pred[val_pred["userId"].isin(worst_pred_users)]

movies_df = pd.read_csv(prefix + "movies-" + data_size + ".csv")
movies_df.set_index("movieId", inplace=True)
movies_df['genres'] = movies_df['genres'].map(lambda x: x.split('|'))

worst_pred = worst_pred.join(other=movies_df, on="movieId", how="inner")
worst_pred = worst_pred.sort_values("sq_loss", ascending=False)

print(worst_pred.head(N_worst))

