start:
	cat product_regression_all_linear_rank_1_depth_8.txt | perl extract.pl

measures.%.txt: product_%.txt
	cat $^ | perl extract2.pl > $@

ALL := $(patsubst product_%.txt,measures.%.txt,$(wildcard product_*.txt))

all: $(ALL)

clean:
	rm -Rf *.measures.txt
	rm -Rf measures.*.txt
