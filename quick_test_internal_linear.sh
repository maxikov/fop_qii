#!/usr/bin/env sh

time spark-submit --driver-memory 5g MovieLensALS.py --checkpoint-dir ~/spark_dir --temp-dir ~spark_dir --spark-executor-memory 5g --local-threads "*" --lmbda 0.2 --num-iter 2 --non-negative --data-path datasets/ml-1m/ --num-partitions 7 --rank 3 --predict-product-features --metadata-sources  genres --drop-rare-features 10 --drop-rare-movies 3 --cross-validation 70 --regression-model linear --filter-data-set 1 --features-trim-percentile 90 --persist-dir ~/quick_test_internal.state > logs/quick_test_internal_linear.txt
