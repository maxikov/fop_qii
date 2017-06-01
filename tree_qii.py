#standard library
import argparse
import os.path
import pickle
import functools

#project files
import rating_explanation
import common_utils

#pyspark libraryb
from pyspark import SparkConf, SparkContext

def get_feature_from_line(line):
        right = line.split("(feature ")[1]
        if " not in " in right:
            left = right.split(" not in ")[0]
        elif "<" in line:
            left = right.split("<")[0]
        elif ">" in line:
            left = right.split(">")[0]
        else:
            left = right.split(" in ")[0]
        fid = int(left)
        return fid

def get_used_features(lr_model):
    model_string = lr_model.toDebugString()
    lines = model_string.split("\n")
    res = set()
    for line in lines:
        if "(feature " not in line:
            continue
        fid = get_feature_from_line(line)
        res.add(fid)
    return res

def load_features_indicators(fname, sc, num_partitions):
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            (training_movies, test_movies, features_test_c, features_training_c,
             features_original_test_c, features_original_training_c,
             indicators_training_c, indicators_test_c) = objects
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            features_training = sc.parallelize(features_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            features_original_training = sc.parallelize(features_original_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                features_test = None
            if features_test_c is not None:
                features_test = sc.parallelize(features_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                features_test = None
            if features_original_test_c is not None:
                features_original_test = sc.parallelize(features_original_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                features_original_test = None
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                indicators_test = None
            return (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test)

def get_feature_mapping(used_features):
    ufl = sorted(list(used_features))
    mapping = {n:f for (n, f) in enumerate(ufl)}
    return ufl, mapping

def get_tree_qii(lr_model, features, used_features):
    ufl, mapping = get_feature_mapping(used_features)
    qiis_list = common_utils.compute_regression_qii(lr_model=lr_model,
                                                    input_features=features,
                                                    target_variable=None,
                                                    logger=None,
                                                    original_predictions=None,
                                                    rank=None,
                                                    features_to_test=used_features)
    qiis_dict = {mapping[f]:q for (f,q) in enumerate(qiis_list)}
    return qiis_dict

def get_top_tree_layers(lr_model):
    model_string = lr_model.toDebugString()
    res = {"root": None, "children": set()}
    for line in model_string.split("\n"):
        if common_utils.beginswith(line, " "*2 +" If") or\
            common_utils.beginswith(line, " "*2 + "Else"):
            fid = get_feature_from_line(line)
            res["children"].add(fid)
        elif common_utils.beginswith(line, " "*1 +" If") or\
            common_utils.beginswith(line, " "*1 + "Else"):
            fid = get_feature_from_line(line)
            res["root"] = fid
    if res["root"] in res["children"]:
        res["children"].remove(res["root"])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--feature", action="store", type=int, help=\
                        "Feature ID of the model to analyze")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)
    print "Loading the tree model"
    lr_model = rating_explanation.load_regression_tree_model(sc, args.persist_dir, args.feature)
    print rating_explanation.get_model_string(lr_model,
            results['feature_names'])
    top_layers = get_top_tree_layers(lr_model)
    print "Tree root:", results["feature_names"][top_layers["root"]], "(",\
        top_layers["root"], ")"
    for child in top_layers["children"]:
        print "Child:", results["feature_names"][child], "(", child, ")"
    print "Loading feature indicator sets"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)

    used_features = get_used_features(lr_model)

    print "Used features:", ", ".join(["{} ({})".format(x, results["feature_names"][x]) for x in used_features])
    print "Computing regression QIIs"
    qiis = get_tree_qii(lr_model, indicators_training, used_features)
    qiis_list = sorted(qiis.items(), key=lambda x: -abs(x[1]))
    for f, q in qiis_list:
        print results["feature_names"][f], "(", f, "):", q

if __name__ == "__main__":
    main()
