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

    header = ["User ID"] + ["Feature {}".format(x) for x in xrange(rank)]
    table = PrettyTable(header)

    for user in args.users:
        ftrs = list(user_features.lookup(user)[0])
        sorted_ftrs = sorted(ftrs, key=lambda x: -abs(x))
        row = [str(user)]
        for ftr in ftrs:
            if ftr == sorted_ftrs[0]:
                row.append(Back.RED + str(ftr) + Style.RESET_ALL)
            elif ftr == sorted_ftrs[1]:
                row.append(Back.YELLOW + str(ftr) + Style.RESET_ALL)
            elif ftr == sorted_ftrs[2]:
                row.append(Back.GREEN + str(ftr) + Style.RESET_ALL)
            elif ftr == sorted_ftrs[-1]:
                row.append(Back.BLUE + str(ftr) + Style.RESET_ALL)
            else:
                row.append(str(ftr))
        table.add_row(row)
    print "Legend:"
    print Back.RED + "Largest magnitude feature" + Style.RESET_ALL
    print Back.YELLOW + "Second largest magnitude feature" + Style.RESET_ALL
    print Back.GREEN + "Third largest magnitude feature" + Style.RESET_ALL
    print Back.BLUE + "Smallest magnitude feature" + Style.RESET_ALL
    print table


if __name__ == "__main__":
    main()
