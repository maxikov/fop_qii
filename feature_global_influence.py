def feature_global_influence(model, rank, user_product_pairs, power=1.0,
        compute_rmse = False, data = None, compute_fast_inf = False):
    original_user_features = model.userFeatures()
    original_product_features = model.productFeatures()
    original_model = (original_user_features, original_product_features)
    print "Computing the predictions of the original model"
    start = time.time()
    original_predictions = manual_predict_all(user_product_pairs,
            *original_model)
    print "Done in", time.time() - start, "seconds"
    res = {"feature_data": {}}
    if compute_rmse:
        print "Computing the mean error of the original model"
        start = time.time()
        error = recommender_mean_error(model, data, power)
        print "Done in", time.time() - start, "seconds"
        print "Mean error:", error
        res["original_rmse"] = error
    for f in xrange(rank):
        res["feature_data"][f] = {}
        if compute_fast_inf:
            pass
        print "Perturbing feature", f, "out of", rank
        start = time.time()
        print "\tPerturbing user feature"
        perturbed_user_features = perturb_feature(original_user_features, f)
        print "\tDone in", time.time() - start, "seconds"
        print "\tPerturbing product feature"
        start = time.time()
        perturbed_product_features = perturb_feature(original_product_features,
                f)
        print "\tDone in", time.time() - start, "seconds"
        perturbed_model = (perturbed_user_features, perturbed_product_features)
        print "\tComputing the predictions of the perturbed model"
        start = time.time()
        perturbed_predictions = manual_predict_all(user_product_pairs,
            *perturbed_model)
        print "\tDone in", time.time() - start, "seconds"
        print "\tComparing models"
        start = time.time()
        predictionsAndRatings = original_predictions.map(lambda x: ((x[0], x[1]), x[2])) \
            .join(perturbed_predictions.map(lambda x: ((x[0], x[1]), x[2]))) \
            .values()
        diff = mean_error(predictionsAndRatings, power)
        print "\tDone in", time.time() - start, "seconds"
        print "\tAverage difference:", diff
        res["feature_data"][f]["model_diff"] = diff
        if compute_rmse:
            print "\tComputing the mean error of the original model"
            start = time.time()
            error = manual_recommender_mean_error(perturbed_user_features,
                    perturbed_product_features, data)
            print "\tDone in", time.time() - start, "seconds"
            print "\tMean error:", error
            res["feature_data"][f]["rmse"] = error
    return res

def internal_feature_influence():
        if sample_type == "training":
            user_product_pairs = training.map(lambda x: (x[0], x[1]))
        elif sample_type == "random":
            all_movies = list(set(training.map(lambda x: x[1]).collect()))
            all_users = list(set(training.map(lambda x: x[0]).collect()))
            pairs = [(random.choice(all_users), random.choice(all_movies)) for\
                    _ in xrange(sample_size)]
            user_product_pairs = sc.parallelize(pairs)
        print "Training model"
        start = time.time()
        model = ALS.train(training, rank, numIter, lmbda)
        print "Done in {} seconds".format(time.time() - start)
        infs = feature_global_influence(model, rank, user_product_pairs, 1.0,
                compute_mean_error, training, compute_fast_influence)
        if compute_mean_error:
            print "Mean error of the original model:", infs["original_rmse"]
        all_features = infs["feature_data"].keys()
        all_features.sort(key=lambda x: -infs["feature_data"][x]["model_diff"])
        title = ["Feature", "Global influence"]
        if compute_mean_error:
            title += ["Perturbed mean error", "delta mean error"]
        table = PrettyTable(title)
        for f in all_features:
            row = [f, infs["feature_data"][f]["model_diff"]]
            if compute_mean_error:
                row += [infs["feature_data"][f]["rmse"],
                        infs["feature_data"][f]["rmse"] -\
                        infs["original_rmse"]]
            table.add_row(row)
        print table
