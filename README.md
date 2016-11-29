# Exploring QII on a Recommender System

18-734 Foundations of Privacy Course Project<br/>
Carnegie Mellon University

Team Members:
* [Jenna MacCarley](https://github.com/jmaccarl)
* [Sophia Kovaleva](https://github.com/maxikov)
* [Dan Calderon](https://github.com/ddcv)

## Setup Instructions:

1) Download latest version of Apache Spark:

http://spark.apache.org/downloads.html

2) Extract & change path of spark folder to /usr/local/spark:

tar xvf spark-2.0.1-bin-hadoop2.7.tgz

mv spark-2.0.1-bin-hadoop2.7  /usr/local/spark

3) Run:

/usr/local/spark/bin/spark-submit MovieLensALS.py datasets/ml-1m/ [arguments]

Use --help for full listing of possible arguments 
