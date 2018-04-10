# IS-ASGD
This is an implementation of importance aampling for ASGD, namely, IS-ASGD.
We recommend the following datasets, the first two datasets are sufficiently large and sparse, 
the last two datasets are small-scale and relative dense. IS-ASGD shows different performance on them.
## Data Preparation
1. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.bz2
2. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kddb.bz2
3. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
4. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2

copy these datasets to 'data' folder and unzip

## Preparation
1. The cal_xnorm.py in script folder is used to calculate the Lipschtz constant for data sample.
2. The cal_random.py in script folder is used to generate random data segmentation.
3. Program reads in norm file and generate sampling distribution at the beginning of each epoch.
4. In fact, the sampling sequence can be generated only once and randomly shuffled for each epoch.

## Run Command
Use the run scripts in 'script' folder

## Visualize the result
Use the print scripts in 'script' folder, IS-ASGD shows better absolute convergence curve than ASGD
and SVRG-ASGD in these large-scale sparse datasets due to sparsity. 

## Testbed
Intel Xeon series is preferred since it has many cores, our testbed is
a two-sockets server of Xeon-2699 V4 CPU.

## Thanks
Jason.y.ye, Intel Asia Pacific R&D Ltd., Shanghai.
Advanced Networking Lab, Shanghai Jiao Tong University, Shanghai.
