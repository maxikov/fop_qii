def correlate_genres(sc, genres, movies, ratings, rank, numIter, lmbda,
        invert_labels=False, no_threshold=False, classifier_model="logistic"):
        print "Bulding per-genre movie lists"
        start = time.time()
        gdict = dict(genres.collect())
        mrdd = sc.parallelize(movies.keys())
        all_genres = sorted(list(genres.map(lambda (_, x): x).fold(set(), lambda x, y:
            set(x).union(set(y)))))
        movies_by_genre = {}
        for cur_genre in all_genres:
            print "Processing {}".format(cur_genre)
            cur_movies = mrdd.map(lambda x: (x, 1 ^ invert_labels
                if cur_genre in gdict[x] else 0 ^ invert_labels))
            movies_by_genre[cur_genre] = dict(cur_movies.collect())
        print "Done in {} seconds".format(time.time() - start)

        print "Training model"
        start = time.time()
        model = ALS.train(ratings, rank, numIter, lmbda)
        print "Done in {} seconds".format(time.time() - start)
        features = dict(model.productFeatures().collect())

        print "Building a family of regresions"
        reg_models = {}
        start = time.time()
        avgbetter = 0.0
        for cur_genre, cur_movies in movies_by_genre.items():
            print "Processing {}".format(cur_genre)
            lr_data = [LabeledPoint(lbl, features[mid])
                    for (mid, lbl) in cur_movies.items()
                    if mid in features]
            lr_data = sc.parallelize(lr_data)
            n_pos = lr_data.filter(lambda x: x.label == 1).count()
            prate = float(n_pos)/float(lr_data.count())
            print "Percent of positives: {:3.1f}%".\
                    format(100*prate)
            if classifier_model == "logistic":
                lr_model = pyspark.\
                        mllib.\
                        classification.\
                        LogisticRegressionWithLBFGS.\
                        train(lr_data)
            elif classifier_model == "decision_tree":
                lr_model = pyspark.\
                        mllib.\
                        tree.\
                        DecisionTree.\
                        trainClassifier(lr_data,2,{})
            labels = lr_data.map(lambda x: x.label)
            if no_threshold:
                lr_model.clearThreshold()
                scores = lr_model.predict(lr_data.map(lambda x:
                    x.features))
                predobs = scores.zip(labels).map(
                    lambda(a, b): (float(a), float(b)))
                metrics = BinaryClassificationMetrics(predobs)
                auroc = metrics.areaUnderROC
                aupr = metrics.areaUnderPR
                better = (1.0 - prate)/(1.0 - aupr)
                reg_models[cur_genre] = {"auroc": auroc,
                            "auprc": aupr, "prate": prate, "model": lr_model, "better":
                            better}
                avgbetter += better
                print "Area under ROC: {:1.3f}, area under precision-recall curve: {:1.3f} ".\
                            format(auroc, aupr) +\
                            "(AuPRc for a random classifier: {:1.3f}, {:1.3f} times better)\n".\
                           format(prate, better)
            else:
                predictions = lr_model.predict(lr_data.map(lambda x:
                    x.features))
                predobs = predictions.zip(labels).map(
                    lambda(a, b): (float(a), float(b)))
                corrects = predobs.filter(
                        lambda (x, y): (x == y))
                fp_count = predobs.filter(lambda (x, y):
                        (x == 1) and (y == 0)).count()
                tp_count = corrects.filter(lambda (x, y): (x==1)).count()
                tn_count = corrects.filter(lambda (x, y): (x==0)).count()
                p_count = labels.filter(lambda x: (x==1)).count()
                n_count = labels.filter(lambda x: (x==0)).count()
                total_count = predobs.count()
                acc = float(tp_count + tn_count)/total_count
                print "Accuracy (tp+tn)/(p+n): {:3.1f}%".format(100*acc)
                if p_count > 0:
                    recall = float(tp_count)/p_count
                    print "Recall (sensitivity, tp rate, tp/p): {:3.1f}%".\
                        format(100*recall)
                else:
                    recall = 0
                    print "No positives in the data set, setting recall"+\
                            " (sensitivity, tp rate, tp/p) to 0"
                if n_count > 0:
                    specificity = float(tn_count)/n_count
                    print "Specificity (tn rate, tn/n): {:3.1f}%".\
                            format(100*specificity)
                else:
                    specificity = 0
                    print "No negatives in the data set, setting specificity"+\
                            " (tn rate, tn/n) to 0"
                if tp_count+fp_count > 0:
                    precision = float(tp_count)/(tp_count+fp_count)
                    print "Precision (positive predictive value, tp/(tp+fp)):" +\
                        " {:3.1f}%".format(100*precision)
                else:
                    precision = 0
                    print "No positives predicted by the classifier"+\
                            " (tp+fp <= 0), setting precision ("+\
                            "positive predictive value, tp/(tp+fp)) to 0"
                if tn_count+fp_count > 0:
                    fpr = float(fp_count)/(tn_count + fp_count)
                    print "False positive rate (fp/(tn+fp)): {:3.1f}%".\
                            format(100*fpr)
                else:
                    fpr = 0
                    print "No true negatives of false positives, setting"+\
                            "false positive rate (fp/(tn+fp)) to 0"
                print ""
                avgbetter += recall
                reg_models[cur_genre] = {"total_count": total_count,
                        "tp_count": tp_count, "fp_count": fp_count,
                        "p_count": p_count, "n_count": n_count,
                        "accuracy": acc, "recall": recall,
                        "specificity": specificity, "precision": precision,
                        "fpr": fpr, "model": lr_model}
        avgbetter = avgbetter/float(len(movies_by_genre))
        if no_threshold:
            print avgbetter, "times better than random on average"
        else:
            print "Average recall: {:3.1f}%".format(100*avgbetter)
        print "Done in {} seconds".format(time.time() - start)

        print "{} genres".format(len(reg_models))

        #Trying to bring it closer to diagonal
        reg_models_src = reg_models.items()
        if classifier_model == "logistic":
            reg_models_res = []
            for i in xrange(len(reg_models_src)):
                ind = min(
                        enumerate(
                                abs(i - (min if invert_labels else max)(
                                            enumerate(x["model"].weights),
                                            key = lambda y: y[1]
                                           )[0]
                                   ) for (gnr, x) in reg_models_src
                                 ), key = lambda y: y[1]
                         )[0]
                reg_models_res.append(reg_models_src[ind])
                del reg_models_src[ind]
        else:
            reg_models_res = reg_models_src
        return reg_models_res, avgbetter

def years_correlator():
        print "Loading years"
        start = time.time()
        years = sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseYear)
        print "Done in {} seconds".format(time.time() - start)
        print "Training model"
        start = time.time()
        model = ALS.train(training, rank, numIter, lmbda)
        print "Done in {} seconds".format(time.time() - start)
        print "Preparing features"
        start = time.time()
        features = model.productFeatures()
        data = features.join(years).map(
                lambda (mid, (ftrs, yr)):
                        LabeledPoint(yr, ftrs))
        print "Done in {} seconds".format(time.time() - start)
        if regression_model == "regression_tree":
            lr_model = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainRegressor(
                            data,
                            categoricalFeaturesInfo={},
                            impurity = "variance",
                            maxDepth = int(math.ceil(math.log(nbins, 2))),
                            maxBins = nbins)
        elif regression_model == "linear":
            print "Building linear regression"
            start = time.time()
            lr_model = LinearRegressionWithSGD.train(data)
        print "Done in {} seconds".format(time.time() - start)
        observations = data.map(lambda x: x.label)
        predictions = lr_model.predict(data.map(lambda x:
                x.features))
        predobs = predictions.zip(observations).map(lambda (a, b): (float(a),
            float(b)))
        metrics = RegressionMetrics(predobs)
        print "RMSE: {}, variance explained: {}, mean absolute error: {}".\
                format(metrics.explainedVariance,\
                metrics.rootMeanSquaredError,
                metrics.meanAbsoluteError)
        if regression_model == "linear":
            print "Weights: {}".format(lr_model.weights)
        elif regression_model == "regression_tree":
            print lr_model.toDebugString()

def genres_correlator():
        print "Loading genres"
        start = time.time()
        if "ml-20m" in movieLensHomeDir:
            mv_file = sc.parallelize(loadCSV(join(movieLensHomeDir, "movies.csv")))
            genres = mv_file.map(lambda x: parseGenre(x, sep=","))
        else:
            genres = sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseGenre)
        print "Done in {} seconds".format(time.time() - start)

        if iterate_rank:
            results = []
            for rank in xrange(iterate_from, iterate_to+1, iterate_step):
                print "Processing rank", rank
                start = time.time()
                reg_models_res, avgbetter = correlate_genres(sc, genres, movies,
                        training, rank, numIter, lmbda, invert_labels,
                        no_threshold, classifier_model=classifier_model)
                reg_models_res = dict(reg_models_res)
                results.append({"rank": rank,
                                "reg_models_res": reg_models_res,
                                "avgbetter": avgbetter})
                print "Done in {} seconds".format(time.time() - start)
            genre_averages = defaultdict(lambda: 0.0)
            for datum in results:
                genre_averages["Average of all"] += datum["avgbetter"]
                for genre, d in datum["reg_models_res"].items():
                    if no_threshold:
                        genre_averages[genre] += d["better"]
                    else:
                        genre_averages[genre] += d["recall"]
            genre_averages = {k: v/float(len(results)) for k, v in
                    genre_averages.items()}
            avgall = genre_averages["Average of all"]
            del genre_averages["Average of all"]
            genre_averages_lst = genre_averages.items()
            genre_averages_lst.sort(key=lambda x: -x[1])
            genre_averages_lst = [("Average of all", avgall)] +\
                genre_averages_lst
            title = ["Genre"] + ["rank: {}".format(x["rank"]) for x in results]
            table = PrettyTable(title)
            for cur_genre, avg in genre_averages_lst:
                if no_threshold:
                    row = ["{} (AVG: {:1.3f})".format(cur_genre, avg)]
                    if cur_genre == "Average of all":
                        row += ["{:1.3f}".format(x["avgbetter"]) for x in results]
                    else:
                        row += ["{:3.1f}%".format(
                            x["reg_models_res"][cur_genre]\
                            ["better" if no_threshold else "recall"]*100)
                            for x in results]
                else:
                    row = ["{} (AVG: {:3.1f}%)".format(cur_genre, avg*100)]
                    if cur_genre == "Average of all":
                        row += ["{:3.1f}%".format(x["avgbetter"]*100) for x in results]
                    else:
                        row += ["{:3.1f}%".format(
                            x["reg_models_res"][cur_genre]\
                            ["better" if no_threshold else "recall"]*100)
                            for x in results]
                table.add_row(row)
            table.align["Genre"] = "r"
            print table

        else:
            reg_models_res, avgbetter = correlate_genres(sc, genres, movies,
                    training, rank, numIter, lmbda, invert_labels, no_threshold)

            for cur_genre, d in reg_models_res:
                row = (" "*3).join("{: 1.4f}".format(coeff)
                        for coeff in d["model"].weights)
                if no_threshold:
                    print "{:>12} (AuPRc: {:1.3f}, Prate: {:1.3f}, {:1.3f}x better) {}".\
                            format(cur_genre, d["auprc"], d["prate"], d["better"], row)
                else:
                    print "{:>12} (recall (tp/p): {:3.1f}%) {}".\
                            format(cur_genre, d["recall"]*100, row)
            if no_threshold:
                print "Average recall: {:3.1f}%".format(avgbetter*100)

        if gui:
            if iterate_rank:
                colors = ['k', 'r', 'g', 'b', 'y']
                styles = ['-', '--', '-.']
                markers = ["o", "^", "s"]
                csms = [(c, s, m) for m in markers for s in styles for c in colors]
                fig, ax = plt.subplots()
                ranks = [x["rank"] for x in results]
                for i in xrange(len(genre_averages_lst)):
                    color, style, marker = csms[i]
                    cur_genre, avg = genre_averages_lst[i]
                    if cur_genre == "Average of all":
                        avgs = [x["avgbetter"]*(1 if no_threshold else 100) for x in results]
                        lw = 2
                    else:
                        avgs = [x["reg_models_res"][cur_genre]\
                                ["better" if no_threshold else "recall"]*\
                                (1 if no_threshold else 100)
                                for x in results]
                        lw = 1
                    if no_threshold:
                        line_label = "{} (AVG: {:1.3f})".format(cur_genre, avg)
                    else:
                        line_label = "{} (AVG: {:3.1f}%)".format(cur_genre, avg*100)
                    ax.plot(ranks, avgs, color = color, linestyle=style,
                            label = line_label, marker=marker, lw=lw)
                legend = ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
                ax.set_xticks(ranks)
                ax.set_xticklabels(ranks)
                ax.set_xlabel("Rank")
                if no_threshold:
                    ax.set_ylabel("Quality of {}".format(classifier_model))
                else:
                    if invert_labels:
                        ax.set_ylabel("Inverted recall (tn/n), %")
                    else:
                        ax.set_ylabel("Recall (tp/p), %)")
                ax.set_title("Performance of {} ".format(classifier_model) +\
                     ("with inverted labels " if invert_labels else "") +\
                     "from movie matrix to genres")
                plt.show()
            else:
                matrix = [list(x["model"].weights) for _, x in reg_models_res]
                matrix = numpy.array(matrix)
                fig, ax = plt.subplots()
                cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
                ax.set_yticks(range(len(reg_models_res)))
                if no_threshold:
                    ax.set_yticklabels("{} ({:1.1f}x)".format(x,\
                       d["better"]) for x, d in reg_models_res)
                else:
                    ax.set_yticklabels("{} ({:3.1f}% recall)".format(x,\
                            d["recall"]*100) for x, d in reg_models_res)
                ax.set_ylabel("Genre")
                ax.set_xticks(range(len(reg_models_res[0][1]["model"].weights)))
                ax.set_xticklabels(range(len(reg_models_res[0][1]["model"].weights)))
                ax.set_xlabel("Product Features")
                ax.set_title("Coefficients for logistic regression"+\
                        (" with inverted labels" if invert_labels else "") +\
                        (" ({:1.3f} times better than random on average)"
                            if no_threshold
                            else "({:3.1f}% true positives)").\
                        format(avgbetter * (1 if no_threshold else 100)))
                cbar = fig.colorbar(cax)
                plt.show()