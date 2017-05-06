#!/usr/bin/env sh

time spark-submit --driver-memory 5g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 5g --local-threads "*" --lmbda 0.2 --num-iter 2 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 3 --predict-metadata --metadata-sources years genres tags --drop-rare-features 1000 --drop-rare-movies 4 --cross-validation 70 --regression-model linear --persist-dir ~/quick_test.state --filter-data-set 1 > logs/quick_test.txt
