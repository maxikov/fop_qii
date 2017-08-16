#!/usr/bin/env bash

python explanation_correctness.py --persist-dir /home/sophiak/fop_qii/archived_states/product_regression_all_regression_tree_rank_10_depth_5_larger_synth_noisy_profile_recommender.state --dataset-dir new_experiments/synth_data_set --qii-iterations 20 --sample-size 40 --movies-file datasets/ml-20m/movies.csv > logs/larger_synth_user_profile_noisy_recommender_correctness.txt
