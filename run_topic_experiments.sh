#!/usr/bin/env bash

log_dir="new_experiments/logs"

log_file="${log_dir}/topic_modeling_real_data_${algo}_${model}.txt"
whole_command="spark-submit --driver-memory 10g product_topics.py --persist-dir archived_states/product_regression_all_regression_tree_rank_12_depth_5.state --movies-file datasets/ml-20m/movies.csv >> $log_file"
echo `date` "Running ${whole_command}, writing to ${log_file}"
echo `date` $whole_command > $log_file
eval $whole_command
