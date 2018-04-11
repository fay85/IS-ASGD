// Copyright 2012 Victor Bittorf, Chris Re
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Hogwild!, part of the Hazy Project
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)
// Original Hogwild! Author: Chris Re (chrisre [at] cs.wisc.edu)             
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include "hazy/hogwild/hogwild-inl.h"
#include "hazy/hogwild/memory_scan.h"
#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/binfscan.h"

#include "frontend_util.h"

#include "svm/svmmodel.h"
#include "svm/svm_loader.h"
#include "svm/svm_exec.h"


// Hazy imports
using namespace hazy;
using namespace hazy::hogwild;
using scan::TSVFileScanner;
using scan::MatlabTSVFileScanner;

using hazy::hogwild::svm::fp_type;


using namespace hazy::hogwild::svm;


int use_log;
int main(int argc, char** argv) {
  bool matlab_tsv = false;
  bool loadBinary = false;
  unsigned nepochs = 20;
  //unsigned nthreads = 1;
  double mu = 1.0, step_size = 5e-2, step_decay = 0.8;
  static struct extended_option long_options[] = {
    {"mu", required_argument, NULL, 'u', "the maxnorm"},
    {"epochs"    ,required_argument, NULL, 'e', "number of epochs (default is 20)"},
    {"stepinitial",required_argument, NULL, 'i', "intial stepsize (default is 5e-2)"},
    {"step_decay",required_argument, NULL, 'd', "stepsize decay per epoch (default is 0.8)"},
    {"seed", required_argument, NULL, 'q', "random seed (o.w. selected by time, 0 is reserved)"},
    {"splits", required_argument, NULL, 'r', "number of threads (default is 1)"},
    {"CrossEntropy", required_argument, NULL, 'c', "Use CrossEntropyLoss(default hinge loss)"},
    {"binary", required_argument,NULL, 'v', "load the file in a binary fashion"},
    {"matlab-tsv", required_argument,NULL, 'm', "load TSVs indexing from 1 instead of 0"},
    {"use_IS", required_argument,NULL, 'p', "use IS or not"},
    {"svrg", required_argument,NULL, 'g', "SVRG enable"},
    {"lip", required_argument,NULL, 'l', "lipschitz"},
    {"dis", required_argument,NULL, 't', "lipschitz-distributed"},
    {"shuffled", required_argument,NULL, 's', "Load Lipshitch-shuffled Data"},
    {"random_dis", required_argument,NULL, 'a', "Random dis"},
    {NULL,0,NULL,0,0} 
  };

  char usage_str[] = "<train file> <test file>";
  int c = 0, option_index = 0;
  option* opt_struct = convert_extended_options(long_options);
  while( (c = getopt_long(argc, argv, "", opt_struct, &option_index)) != -1) 
  {
    switch (c) { 
      case 'v':
        loadBinary = (atoi(optarg) != 0);
        break;
      case 'm':
        matlab_tsv = (atoi(optarg) != 0);
        break;
      case 'u':
        mu = atof(optarg);
        break;
      case 'e':
        nepochs = atoi(optarg);
        break;
      case 'i':
        step_size = atof(optarg);
        break;
      case 'd':
        step_decay = atof(optarg);
        break;
      case 'r':
        nthreads = atoi(optarg);
        break;
      case 'p':
        use_prob = atoi(optarg);
        break;
      case 'g':
        svrg = atoi(optarg);
        break;
      case 'l':
        lipschitz = atoi(optarg);
        break;
      case 't':
        distributed= atoi(optarg);
        break;
      case 's':
      	shuffled = atoi(optarg);
      	break;
      case 'a':
      	random_dis = atoi(optarg);
      	break;
	  case 'c':
	    use_log= atoi(optarg);
	    break;
      case ':':
      case '?':
        print_usage(long_options, argv[0], usage_str);
        exit(-1);
        break;
    }
  }
  SVMParams tp (step_size, step_decay, mu);

  char * szTestFile;
  if(optind == argc - 2) {
    szExampleFile = argv[optind];
    szTestFile  = argv[optind+1];
  } else {
    print_usage(long_options, argv[0], usage_str);
    exit(-1);
  }

  if(use_prob){std::cout<<"use IS\n";}
  
  size_t nfeats;
  printf("Loading train file %s\n",szExampleFile);
  nfeats = LoadLibSVMStyleFiles(szExampleFile, train_examps);

  train_ended=0;
  //std::cout<<"Generate IS sequence start\n";
  char *prob="_prob";
  char *prob_lip="_prob_lip";
  char *norm_sort="_norm_sort";
  //prob_file stores the sampling probablity of each sample
  strcpy(prob_file,szExampleFile);
  //data_seg_file stores a sorted data index which is used to indicate the data distribution
  strcpy(data_seg_file,szExampleFile);
  if(lipschitz){
  	strcat(prob_file,prob_lip);
  	if(random_dis){
		strcat(data_seg_file,"_norm_random");
  	} else {
  		strcat(data_seg_file,"_norm_balanced");
  	}
  }else{
  	strcat(prob_file,prob);
  }
  if(use_prob){
  	LoadProbability(prob_file,train_examps);
  	if(distributed){
    	LoadDataSeg(data_seg_file);
    }
  }else{
	  for(unsigned int i=0;i<train_examps.size;i++){
        if(use_prob==0){
            train_examps[i].probability=double(1.0)/train_examps.size;
        }
	  }
  }
  init_thread_dataset(nepochs);
  unsigned *degs = new unsigned[nfeats];
  printf("Sample dimensions %lu, nthreads %d\n", nfeats, nthreads);
  for (size_t i = 0; i < nfeats; i++) {
    degs[i] = 0;
  }
  CountDegrees(train_examps, degs);
  tp.degrees = degs;
  tp.ndim = nfeats;
  degree = *tp.degrees;
  std::cout<<"Degrees "<<*tp.degrees<<std::endl;
  SVMModel m(nfeats);
  hazy::thread::ThreadPool tpool(nthreads);
  // These are sampling thread
  tpool.Init();
  MemoryScan<SVMExample> mscan(train_examps);
  Hogwild<SVMModel, SVMParams, SVMExec>  hw(m, tp, tpool);
  hw.RunExperiment(nepochs, mscan);
  Release_IS_Sequence();
  delete[] degs;
  std::cout<<"Train finished\n";
  exit(0);
}

