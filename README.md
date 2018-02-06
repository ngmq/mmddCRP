# mmddCRP
Maximum Margin distance-dependent Chinese Restaurant Process

Run:

cd data/

wdbc: 

./../bin/ddcrp-gibbs-example -w true -f wdbc.data -s 23 -b 150 -n 1 -i 0 --g 0.969 --C 0.15 --lambda 0.0002 --S 4.0 > log.txt

mnist:

./../bin/ddcrp-gibbs-example -w true -f mnist/mnist2000.data -s 1 -b 1 -n 0 -i 0 --g 3.77 --C 0.3 --lambda 0.002 --S 2.5 > log.txt


