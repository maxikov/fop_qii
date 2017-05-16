#!/usr/bin/env bash

#Spark parameters
MEMORY="15g"
STATE_TMP_DIR_ROOT="~"
LOG_DIR="logs"
LOCAL_THREADS="\"*\""
NUM_PARTITIONS=7

#Data source paths
DATA_PATH="datasets/ml-20m"
MOVIES_FILE="datasets/ml-20m/ml-20m.imdb.medium.csv"
TVTROPES_FILE="datasets/dbtropes/tropes.csv"

#Recommender parameters
RANK=3
LMBDA=0.01
NUM_ITER=300
NON_NEGATIVE="" #Must be empty or --non-negative

#Regression parameters
METADATA_SOURCES="years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes"
CROSS_VALIDATION=70
REGRESSION_MODEL="regression_tree"
NBINS=32
MAX_DEPTH=5
DROP_RARE_FEATURES=250
DROP_RARE_MOVIES=50
NORMALIZE="" #Must be empty or --normalize

function make_commands() {
	LOG_STATE_NAME="product_regression_all_${REGRESSION_MODEL}_rank_${RANK}_depth_${MAX_DEPTH}"
	PERSIST_DIR="${STATE_TMP_DIR_ROOT}/${LOG_STATE_NAME}.state"
	LOG_FILE="${LOG_DIR}/${LOG_STATE_NAME}.txt"

	SPARK_SUBMIT="spark-submit --driver-memory $MEMORY"

	CHECKPOINT_DIR="$STATE_TMP_DIR_ROOT/spark_dir"
	TEMP_DIR="$STATE_TMP_DIR_ROOT/spark_dir"

	ARGS="--spark-executor-memory $MEMORY --local-threads $LOCAL_THREADS --num-partitions $NUM_PARTITIONS"
	ARGS="${ARGS} --checkpoint-dir $CHECKPOINT_DIR --temp-dir $TEMP_DIR --persist-dir $PERSIST_DIR"
	ARGS="${ARGS} --data-path $DATA_PATH --movies-file $MOVIES_FILE --tvtropes-file $TVTROPES_FILE"
	ARGS="${ARGS} --rank $RANK --lmbda $LMBDA --num-iter $NUM_ITER $NON_NEGATIVE"
	ARGS="${ARGS} --predict-product-features --metadata-sources $METADATA_SOURCES"
	ARGS="${ARGS} --drop-rare-features $DROP_RARE_FEATURES --drop-rare-movies $DROP_RARE_MOVIES"
	ARGS="${ARGS} --cross-validation $CROSS_VALIDATION --regression-model $REGRESSION_MODEL --nbins $NBINS $NORMALIZE"

	WHOLE_COMMAND="$SPARK_SUBMIT MovieLensALS.py $ARGS >> $LOG_FILE"
}

function run_until_succeeds() {
	MY_NAME="$REGRESSION_MODEL rank $RANK depth $MAX_DEPTH"
	echo `date` "Running $MY_NAME, writing to $LOG_FILE, saving to $PERSIST_DIR"
	echo $WHOLE_COMMAND > $LOG_FILE
	iteration=0
	_start=$SECONDS
	iteration_start=$SECONDS
	until $WHOLE_COMMAND
	do
		echo `date` "Iteration $iteration of $MY_NAME failed acter $(($SECONDS - $iteration_start)) seconds ($(($SECONDS - $_start)) total), trying again"
		iteration_start=$SECONDS
		iteration=$(($iteration + 1))
	done
	echo `date` "$MY_NAME done after $(($SECONDS - $_start)) seconds"
}

function rank_1_experiments() {
	local RANK=1
	make_commands
	mkdir -p $PERSIST_DIR
	run_until_succeeds
	REFERENCE_MODEL="${PERSIST_DIR}/als_model.pkl"

	local MAX_DEPTH=8
	make_commands
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds

	local NORMALIZE="--normalize"
	local REGRESSION_MODEL="linear"
	make_commands
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds
}


function rank_3_experiments() {
	local RANK=3
	make_commands
	mkdir -p $PERSIST_DIR
	run_until_succeeds
	REFERENCE_MODEL="${PERSIST_DIR}/als_model.pkl"

	local MAX_DEPTH=8
	make_commands
	mkdir -p $PERSIST_DIR
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds

	local NORMALIZE="--normalize"
	local REGRESSION_MODEL="linear"
	make_commands
	mkdir -p $PERSIST_DIR
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds
}

function rank_12_experiments() {
	local RANK=12
	make_commands
	mkdir -p $PERSIST_DIR
	run_until_succeeds
	REFERENCE_MODEL="${PERSIST_DIR}/als_model.pkl"

	local MAX_DEPTH=8
	make_commands
	mkdir -p $PERSIST_DIR
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds

	local NORMALIZE="--normalize"
	local REGRESSION_MODEL="linear"
	make_commands
	mkdir -p $PERSIST_DIR
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds
}

rank_1_experiments
rank_3_experiments
rank_12_experiments
