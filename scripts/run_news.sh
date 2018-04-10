#!/bin/bash
ini_lr=$1
tnum=$2
lrd=$3
ep=$5
m=0.000001

lip_file=../data/news20.binary_prob_lip_"$ini_lr"
norm_file=../data/news20.binary_norm_balanced_"$ini_lr"

./cal_random.py ../data/news20.binary 
./cal_xnorm.py ../data/news20.binary "$ini_lr"

cp ../data/"$lip_file" ../data/news20.binary_prob_lip
cp ../data/"$norm_file" ../data/news20.binary_norm_balanced

# SGD
../bin/svm --splits 1 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --svrg 0 --shuffled 0 --use_IS 0 --CrossEntropy $4 --binary 0 ../data/news20.binary x

# ASGD standard
../bin/svm --splits $2 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --dis 1 --svrg 0 --shuffled 0 --use_IS 0 --CrossEntropy $4 --binary 0 ../data/news20.binary x

# IS-ASGD balanced dis
../bin/svm --splits $2 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --dis 1 --svrg 0 --lip 1 --random_dis 0 --use_IS 1 --CrossEntropy $4 --binary 0 ../data/news20.binary x

# SVRG-ASGD
../bin/svm --splits $2 --stepinitial $ini_lr --step_decay $lrd --mu $m --epochs $ep --dis 1 --svrg 1 --use_IS 0  --binary 0 --CrossEntropy $4 ../data/news20.binary x
