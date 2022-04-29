import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator

spark = SparkSession.builder.appName('part1').getOrCreate()

train_df = spark.read.csv("ratings-small-train.csv", header=True, schema="rowIndex INT, userId INT, movieId INT, rating FLOAT, timestamp LONG")
print(train_df.show())

val_df = spark.read.csv("ratings-small-val.csv", header=True, schema="rowIndex INT, userId INT, movieId INT, rating FLOAT, timestamp LONG")
print(val_df.show())

# Fit the model
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train_df)

# Evaluate the model
#pred = model.transform(val_df)
#print(pred.head())
#print(pred.count())
n_recs = 100
user_recs = model.recommendForAllUsers(n_recs)
print(user_recs.head())

#evaluator = RankingEvaluator(labelCol="rating", predictionCol="prediction")
#print(evaluator.evaluate(pred))
