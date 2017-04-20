ACTIVE = conf1

PYTHON = python

active: $(ACTIVE)

pylint: *.py
	pylint -f parseable -j 4 *.py

modules_check: modules_check.py
	$(PYTHON) modules_check.py

COMMON := --local-threads '4' --num-partitions 4

conf1: *.py
	$(PYTHON) MovieLensALS.py --rank 16 --lmbda 0.02 --num-iter 250 $(COMMON) --output-model latents

clean:
	rm -Rf *.pyc
	rm -Rf latents_users
	rm -Rf latents_products

.PHONY: active ypylint modules_check conf1