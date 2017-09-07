#!/usr/bin/env bash

#Spark parameters
MEMORY="32g"
LOCAL_THREADS="32"
NUM_PARTITIONS="32"

#Recommender parameters
RANK=20
LMBDA=0.1
NUM_ITER=300
NON_NEGATIVE="" #Must be empty or --non-negative

#Regression parameters
METADATA_SOURCES="years genres average_rating imdb_keywords imdb_producer imdb_director tvtropes tags"
METADATA_SOURCES="${METADATA_SOURCES} imdb_year imdb_rating imdb_cast imdb_cinematographer imdb_composer imdb_languages imdb_production_companies imdb_writer"
CROSS_VALIDATION=70
REGRESSION_MODEL="regression_tree"
NBINS=32
MAX_DEPTH=5
DROP_RARE_FEATURES=250
DROP_RARE_MOVIES=25
NORMALIZE="" #Must be empty or --normalize
FEATURE_TRIM_PERCENTILE=0
NO_HT="--no-ht"
OVERRIDE_ARGS=""
COLD_START="--cold-start 25"

NAME_SUFFIX=""

FILTER_DATA_SET="10"
ALS_CROSS_VALIDATION="--als-cross-validation 100"

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
        ARGS="${ARGS} $CSV --data-path $DATA_PATH --movies-file $MOVIES_FILE --tvtropes-file $TVTROPES_FILE"
        ARGS="${ARGS} --rank $RANK --lmbda $LMBDA --num-iter $NUM_ITER $NON_NEGATIVE"
        ARGS="${ARGS} --predict-product-features --metadata-sources $METADATA_SOURCES"
        ARGS="${ARGS} --drop-rare-features $DROP_RARE_FEATURES --drop-rare-movies $DROP_RARE_MOVIES"
        ARGS="${ARGS} --cross-validation $CROSS_VALIDATION --regression-model $REGRESSION_MODEL --nbins $NBINS --max-depth $MAX_DEPTH $NORMALIZE"
        ARGS="${ARGS} --features-trim-percentile $FEATURE_TRIM_PERCENTILE ${NO_HT} ${OVERRIDE_ARGS} ${COLD_START} --filter-data-set ${FILTER_DATA_SET} ${ALS_CROSS_VALIDATION}"

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
	cp -r $REFERENCE_MODEL "${PERSIST_DIR}/"
	run_until_succeeds
}

function run_and_save() {
	make_commands
	mkdir -p $PERSIST_DIR
	run_until_succeeds
	REFERENCE_MODEL="${PERSIST_DIR}/als_model.pkl"
}

ALL_ROOT="/home/sophiak/fop_qii"
mkdir -p "${ALL_ROOT}/hypothesis_testing_piotr"
ALL_ROOT="${ALL_ROOT}/hypothesis_testing_piotr"
mkdir -p "${ALL_ROOT}/logs"
LOG_DIR="${ALL_ROOT}/logs"
mkdir -p "${ALL_ROOT}/states"
STATE_TMP_DIR_ROOT="${ALL_ROOT}/states"
mkdir -p "${ALL_ROOT}/datasets"
DATASET_ROOT="${ALL_ROOT}/datasets"

ORIGINAL_DATA_ROOT="datasets"

MOVIES_FILE="${ORIGINAL_DATA_ROOT}/ml-20m/ml-20m.imdb.set1.csv"
TVTROPES_FILE="${ORIGINAL_DATA_ROOT}/dbtropes/tropes.csv"
CSV="--csv"

ORIGINAL_STATE_DIR="archived_states/product_regression_all_regression_tree_rank_12_depth_5.state"

#Recommender parameters
RANK=3
LMBDA=0.07
NUM_ITER=300
NON_NEGATIVE="" #Must be empty or --non-negative

N_SUBJECTS=10

CONTR_OR_EXPR=""
RAND=""

for SUBJ in `seq 1 ${N_SUBJECTS}`
do
	NAME_SUFFIX="new_synth_subj_${SUBJ}"
	DATA_PATH="${DATASET_ROOT}/${NAME_SUFFIX}"
	echo `date` "Creating dataset in $DATA_PATH, writing to ${LOG_DIR}/synth_data_set_generation_${NAME_SUFFIX}.txt"
	python synth_dataset_generator.py --specific-features "movielens_genre:Romance" "imdb_keywords:husband-wife-relationship" "movielens_genre:Drama" "imdb_keywords:independent-film" "imdb_keywords:murder" "movielens_genre:Comedy" --persist-dir ${ORIGINAL_STATE_DIR} --n-profiles 3 --n-users 2000 --mu 5 --sigma 1  --odir ${DATA_PATH} > ${LOG_DIR}/synth_data_set_generation_${NAME_SUFFIX}.txt
	cp ${ORIGINAL_DATA_ROOT}/ml-20m/tags.csv ${DATA_PATH}/tags.csv
	echo `date` "Done creating data set"

	echo `date` "Building a recommender and a shadow model"
	run_and_save
	echo `date` "Done building a recommender"

	echo `date` "Running correctness explanations, writing to ${LOG_DIR}/explanation_correctness_${NAME_SUFFIX}.txt"
	python explanation_correctness.py --persist-dir $PERSIST_DIR --dataset-dir $DATA_PATH --qii-iterations 10 --sample-size 20 --movies-file $MOVIES_FILE > "${LOG_DIR}/explanation_correctness_${NAME_SUFFIX}.txt"
	echo `date` "Done correctness explanation"
done
