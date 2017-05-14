#!/usr/bin/env bash


#echo `date`
#echo "Doing tree regression rank 1"
#iteration=0
#_start=$SECONDS
#iteration_start=$SECONDS
#until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 1 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 500 --drop-rare-movies 50 --normalize  --persist-dir ~/all_tree_internal_rank_1.state  > logs/internal_regression_all_tree_rank_1.txt
#do
#    echo "Iteration $iteration of tree regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
#    iteration_start=$SECONDS
#    iteration=$(($iteration + 1))
#done
#echo "Rank 1 tree regression done after $(($SECONDS - $_start)) seconds"

#echo `date`
#echo "Doing tree regression rank 3"
#iteration=0
#_start=$SECONDS
#iteration_start=$SECONDS
#until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 3 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 500 --drop-rare-movies 50 --normalize  --persist-dir ~/all_tree_internal_rank_3.state  > logs/internal_regression_all_tree_rank_3.txt
#do
#    echo "Iteration $iteration of tree regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
#    iteration_start=$SECONDS
#    iteration=$(($iteration + 1))
#done
#echo "Rank 3 tree regression done after $(($SECONDS - $_start)) seconds"


#echo `date`
#echo "Doing tree regression depth 8 rank 3"
#iteration=0
#_start=$SECONDS
#iteration_start=$SECONDS
#until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 3 --predict-product-features --metadata-sources years genres average_rating tvtropes imdb_keywords imdb_producer imdb_director tags --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_tree_internal_rank_3_depth_8.state  > logs/internal_regression_all_tree_rank_3_depth_8.txt
#do
#    echo "Iteration $iteration of tree regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
#    iteration_start=$SECONDS
#    iteration=$(($iteration + 1))
#done
#echo "Rank 3 tree depth 8 regression done after $(($SECONDS - $_start)) seconds"

#echo `date`
#echo "Doing linear regression rank 3"
#iteration=0
#_start=$SECONDS
#iteration_start=$SECONDS
#until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 3 --predict-product-features --metadata-sources years genres average_rating  --cross-validation 70 --regression-model linear --nbins 32 --drop-rare-features 500 --drop-rare-movies 2 --normalize --filter-data-set 1 --persist-dir ~/all_linear_internal_rank_3.state  > logs/internal_regression_all_linear.txt
#do
#    echo "Iteration $iteration of linear regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
#    iteration_start=$SECONDS
#    iteration=$(($iteration + 1))
#done
#echo "Rank 3 linear regression done after $(($SECONDS - $_start)) seconds"

#echo `date`
#echo "Doing logistic regression rank 3"
#iteration=0
#_start=$SECONDS
#iteration_start=$SECONDS
#until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 3 --predict-product-features --metadata-sources years genres average_rating tvtropes imdb_keywords imdb_producer imdb_director tags --cross-validation 70 --regression-model logistic --normalize --nbins 32 --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_logistic_internal_rank_3.state  > logs/internal_regression_all_logistic.txt
#do
#    echo "Iteration $iteration of logistic regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
#    iteration_start=$SECONDS
#    iteration=$(($iteration + 1))
#    sleep 10000000
#done
#echo "Rank 3 logistic regression done after $(($SECONDS - $_start)) seconds"

#echo `date`
#echo "Doing tree regression rank 12"
#iteration=0
#_start=$SECONDS
#iteration_start=$SECONDS
#until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_tree_internal.state  > logs/internal_regression_all_tree.txt
#do
#    echo "Iteration $iteration of tree regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
#    iteration_start=$SECONDS
#    iteration=$(($iteration + 1))
#done
#echo "Rank 12 tree regression done after $(($SECONDS - $_start)) seconds"
#echo `date`

echo `date`
iteration=0
_start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 1 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model linear --normalize --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_linear_internal_rank_1_same_as_tree.state > logs/internal_regression_all_linear_rank_1_same_as_tree.txt
do
    echo "Iteration $iteration of linear regression failed after $(($SECONDS - $_start)) ( $(($SECONDS - $iteration_start)) ) total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done

echo "Linear regression rank 1 same as tree done after $(($SECONDS - $_start))"

echo `date`
iteration=0
_start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 3 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model linear --normalize --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_linear_internal_rank_3_same_as_tree.state > logs/internal_regression_all_linear_rank_3_same_as_tree.txt
do
    echo "Iteration $iteration of linear regression failed after $(($SECONDS - $_start)) ( $(($SECONDS - $iteration_start)) ) total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done

echo "Linear regression rank 3 same as tree done after $(($SECONDS - $_start))"

echo `date`
iteration=0
_start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model linear --normalize --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_linear_internal_rank_12_same_as_tree.state > logs/internal_regression_all_linear_rank_12_same_as_tree.txt
do
    echo "Iteration $iteration of linear regression failed after $(($SECONDS - $_start)) ( $(($SECONDS - $iteration_start)) ) total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done

echo "Linear regression rank 12 same as tree done after $(($SECONDS - $_start))"

echo "Doing tree regression deep rank 12"
iteration=0
_start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model regression_tree --nbins 32 --drop-rare-features 500 --drop-rare-movies 50  --max-depth 8 --persist-dir ~/all_tree_internal_deep.state  > logs/internal_regression_all_tree_deep.txt
do
    echo "Iteration $iteration of tree regression failed after $(($SECONDS - $_start)) ($(($SECONDS - $iteration_start)) seconds total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done
echo "Rank 12 tree regression done after $(($SECONDS - $_start)) seconds"
iteration=0
_start=$SECONDS
iteration_start=$SECONDS
until spark-submit --driver-memory 15g MovieLensALS.py --checkpoint-dir /home/maxikov/spark_dir --temp-dir /home/maxikov/spark_dir --spark-executor-memory 15g --local-threads "*" --lmbda 0.02 --num-iter 300 --non-negative --data-path datasets/ml-20m/ --movies-file datasets/ml-20m/ml-20m.imdb.medium.csv --tvtropes-file datasets/dbtropes/tropes.csv --num-partitions 7 --rank 12 --predict-product-features --metadata-sources years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes --cross-validation 70 --regression-model linear --normalize --drop-rare-features 500 --drop-rare-movies 50 --persist-dir ~/all_linear_internal.state > logs/internal_regression_all_linear.txt
do
    echo "Iteration $iteration of linear regression failed after $(($SECONDS - $_start)) ( $(($SECONDS - $iteration_start)) ) total, trying again"
    iteration_start=$SECONDS
    iteration=$(($iteration + 1))
done

echo "Linear regression done after $(($SECONDS - $_start))"
