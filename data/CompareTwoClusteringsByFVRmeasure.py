"""
Input: two clustering files containg multiple clusters, each in one line. Each line contains a list of
indices of data points that belong to that cluster.

Output: 03 measurement scores (F-measure, V-measure and Adjusted Rand Index).
"""

import numpy as np 
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
import argparse
import os


parser = argparse.ArgumentParser(description="Read data and generate ground truth clustering")
parser.add_argument("-c1", "--clustering1", type=str, nargs='?', help="clustering1 file name")
parser.add_argument("-c2", "--clustering2", type=str, nargs='?', help="clustering2 file name")
parser.add_argument("-s", "--savePath", type=str, nargs='?', help="path to save the results")

args = parser.parse_args()
#print args

if args.clustering1 is None or args.clustering2 is None:
    raise Exception("One of the clustering is unspecified")

if args.savePath is not None:
    savePath = args.savePath + "/"
else:
    savePath = "./"

try:
    assignment1 = {}
    assignment2 = {}

    with open(args.clustering1, 'r') as fh:
        cluster_idx = 0
        for line in fh:
            indices = np.fromstring(line, dtype=int, sep=",")
            for idx in indices:
                assignment1[idx] = cluster_idx
            cluster_idx += 1

    with open(args.clustering2, 'r') as fh:
        cluster_idx = 0
        for line in fh:
            indices = np.fromstring(line, dtype=int, sep=",")
            for idx in indices:
                assignment2[idx] = cluster_idx
            cluster_idx += 1

    # Sanity check
    keys1 = sorted(assignment1.keys())
    keys2 = sorted(assignment2.keys())
    assert(keys1 == keys2)

    N11 = 0 # Number of pair (x,y) that are in the same cluster in both clusterings
    N00 = 0 # Number of pair (x,y) that are NOT in the same cluster in both clusterings
    N10 = 0 # Number of pair (x,y) that are in the same cluster in clustering1 but NOT in the same cluster in clustering2
    N01 = 0 # Number of pair (x,y) that are in the same cluster in clustering2 but NOT in the same cluster in clustering1

    clusters1 = []
    clusters2 = []

    for x in keys1:
        for y in keys1:
            if x != y:
                if assignment1[x] == assignment1[y] and assignment2[x] == assignment2[y]:
                    N11 += 1
                if assignment1[x] != assignment1[y] and assignment2[x] != assignment2[y]:
                    N00 += 1
                if assignment1[x] == assignment1[y] and assignment2[x] != assignment2[y]:
                    N10 += 1
                if assignment1[x] != assignment1[y] and assignment2[x] == assignment2[y]:
                    N01 += 1

        clusters1.append(assignment1[x])
        clusters2.append(assignment2[x])

    # See Table I and II in (Achtert, 2012)
    Fmeasure = 2.0 * N11 / (2.0 * N11 + N10 + N01)
    # See (Rosenberg and Hirschberg, 2007)
    Vmeasure = v_measure_score(clusters1, clusters2)
    # See (Hubert and Arabie, 1985)
    AdjustedRandIndex = adjusted_rand_score(clusters1, clusters2)

    filename1 = os.path.basename(args.clustering1)
    filename2 = os.path.basename(args.clustering2)
    outFilename = "scores_" + filename1.split('.')[0] + "_" + filename2.split('.')[0] + ".txt"
    outFilename = savePath + outFilename

    with open(outFilename, 'w') as fh:
        fh.write("Fmeasure = {}\n".format(Fmeasure))
        fh.write("Vmeasure = {}\n".format(Vmeasure))
        fh.write("AdjustedRandIndex = {}\n".format(AdjustedRandIndex))

        print("Fmeasure = {}\n".format(Fmeasure))
        print("Vmeasure = {}\n".format(Vmeasure))
        print("AdjustedRandIndex = {}\n".format(AdjustedRandIndex))

    print( "Scores saved to {}".format(outFilename) )

except Exception as e:
    raise



