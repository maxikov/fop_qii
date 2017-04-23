ACTIVE = conf1

PYTHON = python

active: $(ACTIVE)

pylint: *.py
	pylint -f parseable -j 4 *.py

modules_check: modules_check.py
	$(PYTHON) modules_check.py

SUBMIT := spark-submit --driver-memory 32g

COMMON := --spark-executor-memory 32g --local-threads '2' --num-partitions 8 \
          --checkpoint-dir checkpoint --temp-dir tmp \
	  --data-path ~/data-movies/movielens/ml-20m/ --movies-file ~/data-movies/movielens/ml-20m.imdb.medium.csv

conf1: *.py
	$(SUBMIT) MovieLensALS.py --rank 16 --lmbda 0.02 --num-iter 250 $(COMMON) --output-model latents

clean:
	rm -Rf *.pyc
	rm -Rf latents_users
	rm -Rf latents_movies

.PHONY: active ypylint modules_check conf1