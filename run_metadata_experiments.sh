#!/usr/bin/env sh

iteration=0
start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating tvtropes imdb_keywords imdb_producer imdb_director tags --cross-validation 70 --regression-model linear --drop-rare-features 100 --drop-rare-movies 10 --persist-dir ~/all_linear.state > logs/metadata_regression_all_linear.txt
do
    echo "Iteration $iteration of linear regression failed after $(($SECONDS - $start)) ($(($SECONDS - $iteration_start)) total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done

echo "Linear regression done after $(($SECONDS - $start))"

iteration=0
start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 20g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 20g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-metadata --metadata-sources years genres average_rating tvtropes imdb_keywords imdb_producer imdb_director tags --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 100 --drop-rare-movies 10 --persist-dir ~/all_tree.state  > logs/metadata_regression_all_tree.txt
do
    echo "Iteration $iteration of tree regression failed after $(($SECONDS - $start)) ($(($SECONDS - $iteration_start)) total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done

echo "Tree regression done after $(($SECONDS - $start))"
