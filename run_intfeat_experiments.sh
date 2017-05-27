#!/usr/bin/env bash

#Spark parameters
MEMORY="15g"
STATE_TMP_DIR_ROOT="/home/maxikov"
LOG_DIR="logs"
LOCAL_THREADS="8"
NUM_PARTITIONS="7"

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
FEATURE_TRIM_PERCENTILE=90

NAME_SUFFIX=""

function make_commands() {
	LOG_STATE_NAME="product_regression_all_${REGRESSION_MODEL}_rank_${RANK}"
	if [ "$REGRESSION_MODEL" == "regression_tree" ]
	then
		LOG_STATE_NAME="${LOG_STATE_NAME}_depth_${MAX_DEPTH}"
	fi
	if [ ${FEATURE_TRIM_PERCENTILE} -ne 0 ]
	then
		LOG_STATE_NAME="${LOG_STATE_NAME}_features_trim_percentile_${FEATURE_TRIM_PERCENTILE}"
	fi
	LOG_STATE_NAME="${LOG_STATE_NAME}_${NAME_SUFFIX}"
	PERSIST_DIR="${STATE_TMP_DIR_ROOT}/${LOG_STATE_NAME}.state"

	SPARK_SUBMIT="spark-submit --driver-memory $MEMORY"

	CHECKPOINT_DIR="$STATE_TMP_DIR_ROOT/spark_dir"
	TEMP_DIR="$STATE_TMP_DIR_ROOT/spark_dir"

	ARGS="--spark-executor-memory $MEMORY --local-threads $LOCAL_THREADS --num-partitions $NUM_PARTITIONS"
	ARGS="${ARGS} --checkpoint-dir $CHECKPOINT_DIR --temp-dir $TEMP_DIR --persist-dir $PERSIST_DIR"
	ARGS="${ARGS} --data-path $DATA_PATH --movies-file $MOVIES_FILE --tvtropes-file $TVTROPES_FILE"
	ARGS="${ARGS} --rank $RANK --lmbda $LMBDA --num-iter $NUM_ITER $NON_NEGATIVE"
	ARGS="${ARGS} --predict-product-features --metadata-sources $METADATA_SOURCES"
	ARGS="${ARGS} --drop-rare-features $DROP_RARE_FEATURES --drop-rare-movies $DROP_RARE_MOVIES"
	ARGS="${ARGS} --cross-validation $CROSS_VALIDATION --regression-model $REGRESSION_MODEL --nbins $NBINS --max-depth $MAX_DEPTH $NORMALIZE"
	ARGS="${ARGS} --features-trim-percentile $FEATURE_TRIM_PERCENTILE"

	WHOLE_COMMAND="$SPARK_SUBMIT MovieLensALS.py $ARGS"
}

function run_until_succeeds() {
	MY_NAME="$REGRESSION_MODEL rank $RANK depth $MAX_DEPTH"
	iteration=0
	_start=$SECONDS
	iteration_start=$SECONDS
	LOG_FILE="${LOG_DIR}/${LOG_STATE_NAME}_attempt_${iteration}.txt"
	echo `date` "Running $MY_NAME, writing to $LOG_FILE, saving to $PERSIST_DIR"
	echo $WHOLE_COMMAND > $LOG_FILE
	until $WHOLE_COMMAND >> $LOG_FILE
	do
		echo `date` "Iteration $iteration of $MY_NAME failed acter $(($SECONDS - $iteration_start)) seconds ($(($SECONDS - $_start)) total), trying again"
		iteration_start=$SECONDS
		iteration=$(($iteration + 1))
		LOG_FILE="${LOG_DIR}/${LOG_STATE_NAME}_attempt_${iteration}.txt"
		echo $WHOLE_COMMAND > $LOG_FILE
		echo `date` "Running $MY_NAME, writing to $LOG_FILE, saving to $PERSIST_DIR"
	done
	echo `date` "$MY_NAME done after $(($SECONDS - $_start)) seconds"
}

function copy_and_run() {
	make_commands
	mkdir -p $PERSIST_DIR
	cp -rv $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds
}

function run_and_save() {
	make_commands
	mkdir -p $PERSIST_DIR
	run_until_succeeds
	REFERENCE_MODEL="${PERSIST_DIR}/als_model.pkl"
}

function rank_n_experiments() {
	run_and_save

	local MAX_DEPTH=8
	copy_and_run

	local MAX_DEPTH=4
	local NBINS=16
	copy_and_run

	local NORMALIZE="--normalize"
	local REGRESSION_MODEL="linear"
	copy_and_run
}

MOVIES_FILE="datasets/ml-20m/ml-20m.imdb.set1.csv"
METADATA_SOURCES="years genres average_rating imdb_keywords imdb_producer imdb_director tags tvtropes imdb_year imdb_rating imdb_cast"
METADATA_SOURCES="${METADATA_SOURCES} imdb_cinematographer imdb_composer imdb_languages imdb_production_companies imdb_writer"
FEATURE_TRIM_PERCENTILE=0
NAME_SUFFIX="extra_imdb_metadata"

RANK=1
run_and_save
RANK=3
run_and_save
RANK=12
run_and_save
RANK=40
run_and_save
