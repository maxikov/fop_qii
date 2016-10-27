# Implementing & Approximating QII on a Recommender System

Setup Instructions:

1) Download latest version of Apache Spark:
http://spark.apache.org/downloads.html

2) Extract & change path of spark folder to /usr/local/spark:
tar xvf spark-2.0.1-bin-hadoop2.7.tgz
mv spark-2.0.1-bin-hadoop2.7  /usr/local/spark

3) Run:
/usr/local/spark/bin/spark-submit MovieLensALS.py datasets/ml-1m/ datasets/personalRatings.txt 
