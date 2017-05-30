#standard library
import argparse
import os.path

#project files
import rating_qii

#pyspark library
from pyspark import SparkConf, SparkContext

#prettytable library
from prettytable import PrettyTable

#colorama library
from colorama import Back, Style

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", action="store", type=str, help=\
                        "Path from which to load models to analyze")
    parser.add_argument("--users", action="store", type=int, nargs="+", help=\
                        "User IDs to print features for.")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]").set("spark.default.parallelism",
                                                 4)
    sc = SparkContext(conf=conf)

    als_model = rating_qii.load_als_model(sc, os.path.join(args.state_path,
                                                           "als_model.pkl"))
    user_features = als_model.userFeatures().sortByKey()
    rank = als_model.rank

    header = ["Used ID"] + ["Feature {}".format(x) for x in xrange(rank)]
    table = PrettyTable(header)

    for user in args.users:
        ftrs = list(user_features.lookup(user)[0])
        max_ftr = max(enumerate(ftrs), key=lambda x: abs(x[1]))[0]
        colored_ftrs = [Back.RED + str(val) + Style.RESET_ALL if n == max_ftr
                        else str(val) for (n, val) in enumerate(ftrs)]
        table.add_row([str(user)] + colored_ftrs)

    print table


if __name__ == "__main__":
    main()
