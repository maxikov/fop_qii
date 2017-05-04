#!/usr/bin/env python

import argparse
import pickle
import os.path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", type=str)
    parser.add_argument("--ofile", type=str)
    args = parser.parse_args()

    fname = os.path.join(args.persist_dir, "results.pkl")
    ifile = open(fname, "rb")
    results = pickle.load(ifile)
    ifile.close()

    res_text = "Overall results dict: {}".format(results)
    if args.ofile is None:
        print res_text
    else:
        ofile = open(args.ofile, "w")
        ofile.write(res_text)
        ofile.close()


if __name__ == "__main__":
    main()
