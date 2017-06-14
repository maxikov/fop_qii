#!/usr/bin/env bash

python explanation_correctness.py --persist-dir new_experiments/product_regression_all_regression_tree_rank_12_depth_5_larger_synth.state --dataset-dir new_experiments/synth_data_set --qii-iterations 40 --sample-size 40 --movies-file datasets/ml-20m/movies.csv > logs/larger_synth_correctness.txt
