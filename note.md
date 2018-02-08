mnist:

Does not converge:

```

./../bin/ddcrp-gibbs-example -w true -f mnist2000.data -s 0 -b 1000 -n 100 -i 0 --g 1.0 --C 0.1 --lambda 0.001 --S 1.0 > log.txt

./../bin/ddcrp-gibbs-example -w true -f mnist2000.data -s 0 -b 1000 -n 100 -i 0 --g 1.0 --C 0.1 --lambda 0.001 --S 1.0 > log.txt

./../bin/ddcrp-gibbs-example -w true -f mnist2000.data -s 0 -b 1000 -n 100 -i 0 --g 0.8 --C 0.05 --lambda 0.001 --S 1.0 > log.txt

./../bin/ddcrp-gibbs-example -w true -f mnist2000.data -s 0 -b 1000 -n 100 -i 0 --g 1.0 --C 0.05 --lambda 0.01 --S 1.5 > log.txt

 ./../bin/ddcrp-gibbs-example -w true -f mnist2000.data -s 0 -b 1000 -n 100 -i 0 --g 1.0 --C 0.01 --lambda 0.03 --S 1.5 > log.txt
 
 ./../bin/ddcrp-gibbs-example -w true -f mnist2000.data -s 0 -b 1000 -n 100 -i 0 --g 1.0 --C 0.01 --lambda 0.01 --S 1.0 > log.txt

```


**Parameter gamma**

The hyperparameter gamma in the exponential decay function is extremely important: f(d) = exp(-d / gamma)

- If gamma is too small, then small distances betwen pairs of points get exaggerated (largeer than alpha) and multiple trival clusters with single data points are more likely to be created.

- If gamma is too large, then large distances between pairs of points get reduced (smaller than alpha) and multiple big clusters with scattered data points are more likely to be created.

As a result, gamma is definitely need to be properly tuned. There might be some heuristic methods to compute the maximum and mininum values of gamma, but generally trial and error works best.

Too see the effect of large and small gamma empirically, test the framework with jain_simple dataset ("jain_simple.txt") with the following parameters:

Small gamma:

```
./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 50 -n 50 -a 2 -s 3 -i 1 --g 0.04 --lambda 0 --C 0.03 --S 1.0 > log.txt
```

Large gamma:

```
./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 50 -n 50 -a 2 -s 3 -i 1 --g 40.0 --lambda 0 --C 0.03 --S 1.0 > log.txt
```

In each case, the posterior distribution of clusterings (e.g. how often clusters with 01 table of all points; how often points from two different classes are in the same clusters) clearly indicates the effect of different values of gammas.

**Parameter C**

If C is too large: the feature vector of each tables will fluctuate and the clustering result will not converge to a proper configuration. To see this effect, generate 50 clustering samples with the following parameters:

```
./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 200 -n 50 -a 2 -s 3 -i 1 --g 1.0 --lambda 0 --C 1.0 --S 1.0 > log.txt
```

The 3rd sample ("clustering_0002") achieves the groundtruth-like result:

```
0, 1, 2
3, 4, 5
```

All subsequent samples achieve the same or very similar results with no two points of different classes in the same cluster. However, the 11th sample ("clustering_0010") has the following configuration:

```
0, 2
1, 5
3, 4
```

which is totally unacceptable because point 1 and point 5 are of two different classes. The same thing happens in the 14th sample:

```
0, 1, 2, 4
3, 5
```

here, the first cluster should not contain point 4.

On the other hand, setting C too small leads to the feature vectors of tables stay at "random" state for too long. In this case the dot products will not capture the necessary similarity features in the likelihood function, so the algorithm either takes a long time to converge or fails to converge at all.

**Parameter S**

Essentially, larger S leads to faster convergence. With larger S, the positive similarities (dot product) get enlarged and the negative similarties get reduced. As the Gibbs sampling processes, for each customer, these changes create highly distinctive probabilities of tables connecting to that customer. S shouldn't be too large though for the sake of numerical stability and for avoiding bad luck when the randomly generated feature vector of a new table happen to be similar to some other existing table or some data points. Too large S could also overpower the distances. For example, running on the jain_simple dataset we get the following log:

```
/../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 210 -n 3 -a 2 -s 3 -i 1 --g 1.0 --lambda 0 --C 0.01 --S 3.0 > log.txt
```

```
============= source = 5
=== currently all tables are:
------ table features ------
*** table id: 0; real id: 0 feature vector:  0.680375 -0.329554
*** table id: 1; real id: 1 feature vector: -0.211234  0.536459
*** table id: 2; real id: 2 feature vector:  0.566198 -0.444451
*** table id: 3; real id: 3 feature vector: 0.59688 0.10794
*** table id: 4; real id: 4 feature vector:   0.823295 -0.0452059
*** table id: 5; real id: 5 feature vector:  0.434594 -0.716795
------ table members ------
*** table id: 0; real id: 0
0
*** table id: 1; real id: 1
1
*** table id: 2; real id: 2
2
*** table id: 3; real id: 3
3
*** table id: 4; real id: 4
4
*** table id: 5; real id: 5
prior = -2.70943, ll = 2.66301; p = -0.0464127; exp = 0.954648
prior = -2.30613, ll = -1.69908; p = -4.00522; exp = 0.0182204
prior = -2.16712, ll = 2.5581; p = 0.390977; exp = 1.47842
prior = -0.111823, ll = 1.53843; p = 1.42661; exp = 4.16454
prior = -0.166686, ll = 2.51198; p = 2.34529; exp = 10.4363
prior = -1.09861, ll = 2.71828; p = 1.61967; exp = 5.05143
assign 5 to table numbered 3 real is 3
```

This is the log from the very first burning iteration. The point source 5 is the middle one of the second class. Since point 5 clearly closer to point 3 than point 4, the probability of linking 5 and 3 should be higher than or at least comparable to the probability of linking 5 and 4. However, under the effect of the scaling factor S = 3.0 the probability of linking 5 and 4 is 10.4363 / 4.16454 = 2.5 times higher than the probability of linking 5 and 3 simply because the randomly generated feature vector of table 4 is more similar to that of table 5. Certainly this is not an expected behavior. Empirically, for jain_simple S = 2 works best in combination with other hyperparameters as follows:

```
./../bin/ddcrp-gibbs-example -w true -f jain_simple.txt -b 210 -n 50 -a 2 -s 3 -i 1 --g 1.0 --lambda 0 --C 0.01 --S 2.0 > log.txt
```

In summary, setting S higher than 1.0 might be desirable to make the clustering result more stable and convering faster, but S should be carefully tuned as well to avoid unexpected effects.

