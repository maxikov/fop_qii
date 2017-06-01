#standard library
import argparse
import os.path

#project files
import rating_explanation
import rating_qii

#pyspark libraryb
from pyspark import SparkConf, SparkContext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--movie", action="store", type=int, help=\
                        "Movie for which to compute the QII")
    parser.add_argument("--user", action="store", type=int, help=\
                        "User for who to compute QII")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)

    print "Original recommender relative to ground truth on training set mean absolute error: {}, RMSE: {}"\
            .format(results["baseline_rec_eval"]["mean_abs_err"],results["baseline_rec_eval"]["rmse"])
    print "Shadow model relative to the baseline recommender on test set MAE: {}, RMSE: {}"\
            .format(results["all_replaced_rec_eval_test"]["mean_abs_err"],results["all_replaced_rec_eval_test"]["rmse"])
    print "Randomized model relative to the baseline recommender on test set MAE: {}, RMSE: {}"\
            .format(results["all_random_rec_eval_test"]["mean_abs_err"],results["all_random_rec_eval_test"]["rmse"])
    print "Shadow model is {} times better than random on the test set"\
            .format(results["all_random_rec_eval_test"]["mean_abs_err"]/results["all_replaced_rec_eval_test"]["mean_abs_err"])

    print "Loading ALS model"
    model = rating_qii.load_als_model(sc, os.path.join(args.persist_dir,
                                                       "als_model.pkl"))

if __name__ == "__main__":
    main()
