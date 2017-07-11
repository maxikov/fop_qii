#!/usr/bin/env bash

#algos="kmeans spectral agglomerative"
algos="kmeans"
#models="mlpc decision_tree"
models="decision_tree"
log_dir="new_experiments/logs"
#n_or_max="--n-clusters 15"
n_or_max="--max-clusters 30"

for algo in $algos
do
	for model in $models
	do
		log_file="${log_dir}/product_clustering_real_data_${algo}_${model}.txt"
		whole_command="spark-submit --driver-memory 10g product_clustering.py --persist-dir archived_states/product_regression_all_regression_tree_rank_12_depth_5.state ${n_or_max} --cluster-model ${algo} --model ${model} --movies-file datasets/ml-20m/movies.csv >> $log_file"
		echo `date` "Running ${whole_command}, writing to ${log_file}"
		echo `date` $whole_command > $log_file
		eval $whole_command
		log_file="${log_dir}/product_clustering_larger_synth_${algo}_${model}.txt"
		whole_command="spark-submit --driver-memory 10g product_clustering.py --persist-dir new_experiments/product_regression_all_regression_tree_rank_12_depth_5_larger_synth.state/ --user-profiles new_experiments/synth_data_set/profiles.pkl ${n_or_max} --cluster-model ${algo} --model ${model} --movies-file datasets/ml-20m/movies.csv >> $log_file"
		echo `date` "Running ${whole_command}, writing to ${log_file}"
		echo `date` $whole_command > $log_file
		eval $whole_command
	done
done
echo `date` "All done"
