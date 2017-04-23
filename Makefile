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

clean:
	rm -Rf *.pyc
	rm -Rf latents_users
	rm -Rf latents_movies

.PHONY: active ypylint modules_check conf1