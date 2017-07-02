#!/usr/bin/env bash

algos="kmeans gmm spectral agglomerative birch"
log_dir="new_experiments/logs"

for algo in $algos
do
	log_file="${log_dir}/product_clustering_real_data_${algo}.txt"
	whole_command="spark-submit --driver-memory 10g product_clustering.py --persist-dir archived_states/product_regression_all_regression_tree_rank_12_depth_5.state  --n-clusters 15 --cluster-model ${algo} --movies-file datasets/ml-20m/movies.csv >> $log_file"
	echo `date` "Running ${whole_command}, writing to ${log_file}"
	echo `date` $whole_command > $log_file
	eval $whole_command
	log_file="${log_dir}/product_clustering_larger_synth_${algo}.txt"
	whole_command="spark-submit --driver-memory 10g product_clustering.py --persist-dir new_experiments/product_regression_all_regression_tree_rank_12_depth_5_larger_synth.state/ --user-profiles new_experiments/synth_data_set/profiles.pkl --n-clusters 15 --cluster-model ${algo} --movies-file datasets/ml-20m/movies.csv >> $log_file"
	echo `date` "Running ${whole_command}, writing to ${log_file}"
	echo `date` $whole_command > $log_file
	eval $whole_command
done
echo `date` "All done"
