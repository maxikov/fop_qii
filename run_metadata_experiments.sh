#!/usr/bin/env sh

#until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 200 --non-negative --data-path datasets/ml-1m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources users --cross-validation 70 --regression-model regression_tree --nbins 32 > logs/metadata_regression_users_toy_size_tree.txt
#do
#  echo "Try again"
#  sleep 100000000000000000000
#done

#echo "First done"

#until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 200 --non-negative --data-path datasets/ml-1m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources users --cross-validation 70 --regression-model linear > logs/metadata_regression_users_toy_size_linear.txt
#do
#  echo "Try again"
#  sleep 1000000
#done

#echo "Second done"

#until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 200 --non-negative --data-path datasets/ml-1m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating --cross-validation 70 --regression-model regression_tree --nbins 8 > logs/metadata_regression_products_toy_size_tree.txt
#do
#  echo "Try again"
#done

#echo "Third done"

#until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 200 --non-negative --data-path datasets/ml-1m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating --cross-validation 70 --regression-model linear > logs/metadata_regression_products_toy_size_linear.txt
#do
#  echo "Try again"
#  sleep 70000
#done

#echo "Fourth done"


until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating tvtropes --cross-validation 70 --regression-model linear --drop-rare-features 10  --persist-dir logs/metadata_regression_products_tropes_linear.state > logs/metadata_regression_products_tropes_linear.txt
do
  echo "Try again"
done

echo "Fifth done"

until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating  tvtropes --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 10  --persist-dir logs/metadata_regression_products_tropes_tree.state  > logs/metadata_regression_products_tropes_tree.txt
do
  echo "Try again"
done

echo "Sicth done"

