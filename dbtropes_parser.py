#!/usr/bin/env python

import argparse
from collections import defaultdict as dd

import pandas

def add_spaces(src):
    res = []
    for i in xrange(len(src)):
        if i == 0:
            res.append(src[i])
        else:
            if src[i].isupper() and res[-1] != " ":
                res.append(" ")
            res.append(src[i])
    return "".join(res)

def fetch_name(src):
    spl = src.split("/")
    if "int_" in spl[-1]:
        res = spl[-2]
    else:
        res = spl[-1].split(">")[0]
    res = preprocess_name(res)
    return res

def preprocess_name(name):
    res = add_spaces(name).lower()
    res = res.split(" (")[0]
    if res[:4] == "the ":
        res = res[4:]
    if res[-5:] == ", the":
        res = res[:-5]
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tropesfname", type=str, nargs=1)
    parser.add_argument("moviesfname", type=str, nargs=1)
    parser.add_argument("ofile", type=str, nargs=1)
    args = parser.parse_args()
    fname = args.tropesfname[0]
    moviesfname = args.moviesfname[0]
    ofile = args.ofile[0]

    media_classes = ["Film", "WesternAnimation",
                     "Series", "Anime"]
    res = []
    cur_line = 0
    all_media = set()
    all_tropes = set()
    with open(fname, "r") as f:
        for line in f:
            cur_line += 1
            sp = line.split("> <")
            if len(sp) != 3:
                continue
            sub, pred, obj = sp
            for mc in media_classes:
                if mc in sub:
                    break
            else:
                continue
            if "/Main" not in obj:
                continue
            trope = fetch_name(obj)
            media = fetch_name(sub)
            res.append({"line": line,
                "media": media, "trope": trope})
            all_media.add(media)
            all_tropes.add(trope)
            cur_line += 1
            if cur_line % 10000 == 0:
                print ("Processing line {}"+\
                        " of 21057602 ({}% done"+\
                        "), "+\
                        " {} entries saved").\
                        format(cur_line, 100*cur_line/21057602, len(res))
    print len(res), "lines loaded"
    print len(all_media), "media loaded"
    print len(all_tropes), "tropes loaded"
    print "Cleaning media from tropes"
    for i in xrange(len(res)-1, -1, -1):
        if res[i]["trope"] in all_media:
            del res[i]
    print len(res), "records remaining"
    print "Building per-movie trope lists"
    movies_tropes = dd(set)
    for r in res:
        movies_tropes[r["media"]].add(r["trope"])
    print "done"

    print "Loading movies"
    movies_df = pandas.read_csv(moviesfname)
    movies_dict = movies_df.to_dict()
    movies_dict["title_lower_spaced"] =\
        {mid: preprocess_name(movies_dict["title"][mid])
                for mid in movies_dict["movieId"].keys()}
    print len(movies_dict["movieId"]), "movies loaded"

    print "Linking records"
    mids_tropes = {}
    for cur_id in movies_dict["movieId"].keys():
        mid = movies_dict["movieId"][cur_id]
        mname = movies_dict["title"][cur_id]
        mname_lower = movies_dict["title_lower_spaced"][cur_id]
        if mname_lower in movies_tropes:
            mids_tropes[mid] = (mname, movies_tropes[mname_lower])
    print len(mids_tropes), "records linked"

    res_arr = [[mid, mname, "|".join(str(t) for t in trs)] for (mid, (mname,
        trs)) in mids_tropes.items()]
    res_df = pandas.DataFrame(res_arr, columns=["movieId",
        "title", "tropes"]).set_index("movieId")
    print res_df
    res_df.to_csv(ofile)

if __name__ == "__main__":
    main()
