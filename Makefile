ACTIVE = conf1

PYTHON = python

active: $(ACTIVE)

pylint: *.py
	pylint -f parseable -j 4 *.py

modules_check: modules_check.py
	$(PYTHON) modules_check.py

SUBMIT := spark-submit --driver-memory 32g

DATAS := --data-path ~/data-movies/movielens/ml-20m/ --movies-file ~/data-movies/movielens/ml-20m.imdb.medium.csv

COMMON := --spark-executor-memory 32g --local-threads '2' --num-partitions 8 \
          --checkpoint-dir checkpoint --temp-dir tmp 
conf1: *.py
	$(SUBMIT) MovieLensALS.py --rank 16 --lmbda 0.02 --num-iter 250 $(COMMON) $(DATAS) --output-model latents

bench1: *.py
	$(SUBMIT) MovieLensALS.py --checkpoint-dir checkpoint --temp-dir tmp --spark-executor-memory 32g --local-threads "*" \
	--lmbda 0.2 --num-iter 2 --num-partitions 8 --rank 3 --predict-product-features --metadata-sources years genres \
	--regression-model naive_bayes --nbins 32 --cross-validation 70 \
	$(DATAS) > a.txt

SPARK_DIR = spark_dir

$(SPARK_DIR):
	mkdir $(SPARK_DIR)

$(SPARK_DIR)/%:
	mkdir $@

quick_test: $(SPARK_DIR) $(SPARK_DIR)/tmp $(SPARK_DIR)/checkpoint $(SPARK_DIR)/quick_test.state *.py
	time spark-submit --driver-memory 5g MovieLensALS.py \
	--checkpoint-dir $(SPARK_DIR)/checkpoint --temp-dir $(SPARK_DIR)/tmp \
	--spark-executor-memory 5g --local-threads "*" \
	--lmbda 0.2 --num-iter 2 --non-negative \
	--data-path datasets/ml-1m/ --num-partitions 7 --rank 3 \
	--predict-product-features --metadata-sources years genres \
	--drop-rare-features 10 --drop-rare-movies 3 --cross-validation 70 \
	--regression-model regression_tree --persist-dir $(SPARK_DIR)/quick_test.state \
	--filter-data-set 1 --features-trim-percentile 90 \
	> logs/quick_test.txt

PROOT := /Users/piotrm/Dropbox/icdm_experiments/piotrs_experiments/states
PDIR := $(PROOT)/product_regression_all_regression_tree_rank_12_depth_5_level0.seed0.state

USER := 0

qii_test:
	python shadow_model_qii.py --output qii.all.$(USER).tsv --persist-dir $(PDIR) --user $(USER) --all-movies --qii-iterations 1 > qii.all.$(USER).txt

clean:
	rm -Rf *.pyc
	rm -Rf latents_users
	rm -Rf latents_movies

.PHONY: active ypylint modules_check conf1