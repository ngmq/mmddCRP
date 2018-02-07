== Parameter gamma ==
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

== Parameter C ==

If C is too large: the feature vector of each tables will fluctuate and the clustering result will not converge to a proper configuration. To see this effect, generate 50 clustering samples with the following parameters:

./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 200 -n 50 -a 2 -s 3 -i 1 --g 1.0 --lambda 0 --C 1.0 --S 1.0 > log.txt

The 3rd sample ("clustering_0002") achieves the groundtruth-like result:

0, 1, 2
3, 4, 5

All subsequent samples achieve the same or very similar results with no two points of different classes in the same cluster. However, the 11th sample ("clustering_0010") has the following configuration:

0, 2
1, 5
3, 4

which is totally unacceptable because point 1 and point 5 are of two different classes. The same thing happens in the 14th sample:

0, 1, 2, 4
3, 5

here, the first cluster should not contain point 4.

On the other hand, setting C too small leads to the feature vectors of tables stay at "random" state for too long. In this case the dot products will not capture the necessary similarity features in the likelihood function, so the algorithm either takes a long time to converge or fails to converge at all.

== Parameter S ==




