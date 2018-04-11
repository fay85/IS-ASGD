#!/bin/bash

ini_lr=$1
tnum=$2
lrd=$3
ep=$5
m=0.000001

lip_file=kddb_prob_lip_"$ini_lr"
python ./cal_xnorm.py ../data/kddb "$ini_lr"
python ./cal_random.py ../data/kddb

cp ../data/"$lip_file" ../data/kddb_prob_lip


# SGD
../bin/svm --splits 1 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --svrg 0 --shuffled 0 --CrossEntropy $4 --use_IS 0 --binary 0 ../data/kddb x

# ASGD standard
../bin/svm --splits $2 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --dis 1 --svrg 0 --shuffled 0 --CrossEntropy $4 --use_IS 0 --binary 0 ../data/kddb x

# IS-ASGD
../bin/svm --splits $2 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --dis 1 --random_dis 1 --svrg 0 --CrossEntropy $4 --lip 1 --use_IS 1 --binary 0 ../data/kddb x

# SVRG 
# very slow, we did not wait days to let it finish
#bin/svm --splits $2 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --dis 1 --CrossEntropy $4 --svrg 1 --use_IS 0  --binary 0 ../data/kddb x


