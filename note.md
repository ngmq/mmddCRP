The hyperparameter gamma in the exponential decay function is extremely important: f(d) = exp(-d / gamma)

- If gamma is too small, then small distances betwen pairs of points get exaggerated (largeer than alpha) and multiple trival clusters with single data points are more likely to be created.

- If gamma is too large, then large distances between pairs of points get reduced (smaller than alpha) and multiple big clusters with scattered data points are more likely to be created.

As a result, gamma is definitely need to be properly tuned. There might be some heuristic methods to compute the maximum and mininum values of gamma, but generally trial and error works best.

Too see the effect of large and small gamma empirically, test the framework with jain_simple dataset ("jain_simple.txt") with the following parameters:

Small gamma:

./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 50 -n 50 -a 2 -s 3 -i 1 --g 0.04 --lambda 0 --C 0.03 --S 1.0 > log.txt

Large gamma:

./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 50 -n 50 -a 2 -s 3 -i 1 --g 40.0 --lambda 0 --C 0.03 --S 1.0 > log.txt

In each case, the posterior distribution of clusterings (e.g. how often clusters with 01 table of all points; how often points from two different classes are in the same clusters) clearly indicates the effect of different values of gammas.
