#standard library
import argparse

#project files
import common_utils
import rating_explanation

#pyspark library
from pyspark import SparkConf, SparkContext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", action="store", type=str, help=\
                        "Path from which to load models to analyze")
    parser.add_argument("--feature", action="store", type=int, help=\
                        "Feature ID for which the model should be displayed")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]").set("spark.default.parallelism",
                                                 4)
    sc = SparkContext(conf=conf)

    results_dict = rating_explanation.load_results_dict(args.state_path)
    feature_names = results_dict["feature_names"]

    lr_model = rating_explanation.load_regression_tree_model(sc,
            args.state_path, args.feature)

    model_string = rating_explanation.get_model_string(lr_model,
            results_dict['feature_names'])

    print model_string

if __name__ == "__main__":
    main()
