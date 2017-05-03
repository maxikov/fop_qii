#!/usr/bin/env sh

until spark-submit --driver-memory 6g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 6g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating tvtropes --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 10   > logs/metadata_regression_products_tropes_tree.txt
do
  echo "Try again"
done

echo "Sicth done"

