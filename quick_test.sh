#!/usr/bin/env sh

time spark-submit --driver-memory 5g MovieLensALS.py --checkpoint-dir ~/spark_dir --temp-dir ~/spark_dir --spark-executor-memory 5g --local-threads "*" --lmbda 0.2 --num-iter 2  --data-path datasets/ml-20m/ --num-partitions 7 --rank 3 --predict-product-features --metadata-sources years genres --drop-rare-features 10 --drop-rare-movies 3 --cross-validation 70 --regression-model regression_tree --persist-dir ~/quick_test.state --filter-data-set 1 --features-trim-percentile 90 --max-depth 8 > logs/quick_test.txt
