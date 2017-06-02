#!/usr/bin/env bash

python explanation_correctness.py --persist-dir archived_states/product_regression_all_regression_tree_rank_12_depth_5_new_synth.state --dataset-dir datasets/new_synth --qii-iterations 20 --sample-size 20 --movies-file datasets/ml-20m/movies.csv > logs/new_synth_correctness.txt
