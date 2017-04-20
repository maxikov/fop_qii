ACTIVE = conf1

PYTHON = python

active: $(ACTIVE)

pylint: *.py
	pylint -f parseable -j 4 *.py

modules_check: modules_check.py
	$(PYTHON) modules_check.py

COMMON := --local-threads '*' --num-partitions 4

conf1: *.py
	$(PYTHON) MovieLensALS.py --rank 4 --lmbda 0.02 --num-iter 10 $(COMMON) --output-model latents

clean:
	rm -Rf *.pyc

.PHONY: active ypylint modules_check conf1