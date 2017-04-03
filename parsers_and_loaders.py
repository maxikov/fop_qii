#standard library
import csv
import StringIO
import traceback

def loadCSV(fname, remove_first_line=True):
    """
    Probably won't work with very large data sets.
    """
    with open(fname, "r") as f:
        res = f.readlines()
        if remove_first_line:
            res = res[1:]
        return res

def parseTag(line, sep="::"):
    """
    Parses a tag record in MovieLens format.
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    x = r.next()
    return (int(x[0]), int(x[1]), x[2])

def parseRating(line, sep="::"):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseIMDBKeywords(line, sep="::"):
    """
    Parses movie genres in format
    movieId,title,genres,imdbId,localImdbID,tmdbId,imdb_genres,imdb_keywords
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    keywords = fields[7]
    keywords = keywords.split("|")
    return mid, set(keywords)

def parseIMDBGenres(line, sep="::"):
    """
    Parses movie genres in format
    movieId,title,genres,imdbId,localImdbID,tmdbId,imdb_genres,imdb_keywords
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    keywords = fields[6]
    keywords = keywords.split("|")
    return mid, set(keywords)

def parseGenre(line, sep="::"):
    """
    Parses movie genres in MovieLens format
    movieId::movieTitle::movieGenre1[|movieGenre2...]
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    print "Genre!"
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    genres = fields[2]
    genres = genres.split("|")
    return mid, set(genres)

def parseYear(line, sep="::"):
    """
    Parses movie years in MovieLens format
    movieId::movieTitle (movieYear)::movieGenre1[|movieGenre2...]
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    mtitle = fields[1]
    try:
        year = mtitle.split("(")[-1]
        year = year.split(")")[0]
        year = int(year)
        return mid, year
    except Exception as e: #Dirty hack, but this isn't even supposed to happen!
        print e
        traceback.print_exc()
        print "mid:", mid
        print "Setting the year to 2000, and continuing"
        return mid, 2000


def parseMovie(line, sep="::"):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    return int(fields[0]), fields[1]

def parseUser(line, sep="::"):
    """
    Parses user metadata from the 1m format:
    UserID::Gender::Age::Occupation::Zip-code

    All demographic information is provided voluntarily by the users and is
    not checked for accuracy.  Only users who have provided some demographic
    information are included in this data set.

    - Gender is denoted by a "M" for male and "F" for female
    - Age is chosen from the following ranges:

        *  1:  "Under 18"
        * 18:  "18-24"
        * 25:  "25-34"
        * 35:  "35-44"
        * 45:  "45-49"
        * 50:  "50-55"
        * 56:  "56+"

    - Occupation is chosen from the following choices:

        *  0:  "other" or not specified
        *  1:  "academic/educator"
        *  2:  "artist"
        *  3:  "clerical/admin"
        *  4:  "college/grad student"
        *  5:  "customer service"
        *  6:  "doctor/health care"
        *  7:  "executive/managerial"
        *  8:  "farmer"
        *  9:  "homemaker"
        * 10:  "K-12 student"
        * 11:  "lawyer"
        * 12:  "programmer"
        * 13:  "retired"
        * 14:  "sales/marketing"
        * 15:  "scientist"
        * 16:  "self-employed"
        * 17:  "technician/engineer"
        * 18:  "tradesman/craftsman"
        * 19:  "unemployed"
        * 20:  "writer"
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    fields[1] = 0 if fields[1] == "M" else 1
    fields[4] = fields[4].split("-")[0]
    fields = map(int, fields)
    return tuple(fields)

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def load_average_ratings(src_rdd):
    ratings = src_rdd.map(lambda (x, y): (x, [y]))
    nof = 1
    cfi = {}
    return (ratings, nof, cfi)

def load_years(src_rdd, sep=","):
    years = src_rdd\
            .map(lambda x: parseYear(x, sep=sep))\
            .map(lambda (x, y): (x, [y]))
    nof = 1
    cfi = {}
    return (years, nof, cfi)

def load_genres(src_rdd, sep=",", parser_function=parseGenre):
    genres = src_rdd.map(lambda x: parser_function(x, sep=sep))
    all_genres = sorted(
        list(
            genres\
                .map(lambda (_, x): x)\
                .fold(
                    set(),
                    lambda x, y: set(x).union(set(y)))))
    indicators_genres = genres.map(
        lambda (mid, cur_genres): (
            mid, map(lambda g: int(g in cur_genres), all_genres)))
    nof = len(all_genres)
    cfi = {x: 2 for x in xrange(nof)}
    return (indicators_genres, nof, cfi)

def load_tags(src_rdd, sep=","):
    tags = src_rdd.map(lambda x: parseTag(x, sep=sep))
    all_tags = set(tags.map(lambda x: x[2]).collect())
    all_tags = sorted(list(all_tags))
    tags = tags\
            .groupBy(lambda x: (x[1], x[2]))\
            .map(lambda x: x[0])\
            .groupBy(lambda x: x[0])\
            .map(lambda (mid, data): (mid, set(d[1] for d in data)))
    indicators = tags.map(
        lambda (mid, cur_tags): (
            mid, map(lambda t: int(t in cur_tags), all_tags)))
    nof = len(all_tags)
    cfi = {x: 2 for x in xrange(nof)}
    return (indicators, nof, cfi)

def load_users(src_rdd, sep=","):
    users = src_rdd\
            .map(lambda x: parseUser(x, sep=sep))\
            .map(lambda x: (x[0], x[1:]))
    nof = 4
    cfi = {}
    return (users, nof, cfi)
