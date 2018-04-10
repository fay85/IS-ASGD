// Copyright 2012 Chris Re, Victor Bittorf
//
 //Licensed under the Apache License, Version 2.0 (the "License");
 //you may not use this file except in compliance with the License.
 //You may obtain a copy of the License at
 //    http://www.apache.org/licenses/LICENSE-2.0
 //Unless required by applicable law or agreed to in writing, software
 //distributed under the License is distributed on an "AS IS" BASIS,
 //WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 //See the License for the specific language governing permissions and
 //limitations under the License.

// The Hazy Project, http://research.cs.wisc.edu/hazy/
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)

#ifndef HAZY_HOGWILD_HOGWILD_INL_H
#define HAZY_HOGWILD_HOGWILD_INL_H
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <sys/mman.h>
#include <cstdio>
#include <boost/filesystem.hpp>
#include "hazy/util/clock.h"
#include "hazy/hogwild/freeforall-inl.h"
#include "../../../../src/svm/svmmodel.h"
#include <omp.h>
#include <cassert>
#include <iomanip>
// See for documentation
#include "hazy/hogwild/hogwild.h"
char *szExampleFile, prob_file[40], data_seg_file[40];
int* random_seq, degree, sample_count;
volatile int train_ended;
volatile double variance=0;
int distributed = 0;
int random_dis = 0;
volatile int eidx;
std::vector<std::string> split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

__inline__ void swap (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void randomize ( int arr[], int n, int seed )
{
    // Use a different seed value so that we don't get same
    // result each time we run this program
    srand ( time(NULL)+seed );
 
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = n-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i+1);
 
        // Swap arr[i] with the element at random index
        swap(&arr[i], &arr[j]);
    }
}



namespace hazy {
namespace hogwild {
	
namespace svm {

	vector::FVector<SVMExample> train_examps;

    void Release_IS_Sequence(void){
        for(int i=0;i<nthreads;i++){
        		delete[] IS_seq[i]; 
				delete[] IS_dataset[i];
        }
        delete[] random_seq;
        delete[] train_examps.values;
    }
    
	void init_thread_dataset(void){
        int len=train_examps.size/nthreads;
        int left=train_examps.size%nthreads;
        int start,end;
        sample_count = train_examps.size;
        for(int i=0;i<nthreads;i++){
            start=i*len;
            end=len*(i+1)+(i==(nthreads-1))*left;
            len=end-start;
            int *seq= new int[len];
			int *dataset= new int[len];
            for(int count=0;count<len;count++){
              seq[count]=0;
			  dataset[count]=0;
            }
			IS_dataset.push_back(dataset);
			IS_seq.push_back(seq);
        }
		
		random_seq=new int[train_examps.size];
		for(unsigned int i=0;i<train_examps.size;i++){
			random_seq[i]=i;
		}
	}

    void Generate_Thread_Dataset(void){
        int len=train_examps.size/nthreads;
        int left=train_examps.size%nthreads;
        int start,end;
        int *dataset;
        for(int i=0;i<nthreads;i++){
            start=i*len;
            end=len*(i+1)+(i==(nthreads-1))*left;
            len=end-start;
            dataset= IS_dataset[i];
            for(int count=0;count<len;count++){
              dataset[count]=random_seq[start+count];
            }
        }
		//std::cout<<"Generate thread dataset finished\n";
        return;  
    }

  void Construct_IS_Distribution(void) {
    int len=train_examps.size;
    double prob_total=0.0,prob_tmp=0.0;    
    int *seq,idx;
    seq=random_seq;
    prob_tmp=0.0;
    prob_total=0.0;
    for(int j = 0;j<len;j++){
        idx=j;
        train_examps[idx].p_s=prob_tmp;
        train_examps[idx].p_e=train_examps[idx].p_s+train_examps[idx].probability;
        train_examps[idx].update_weight=train_examps[idx].probability*len;
        prob_tmp+=train_examps[idx].probability;
        if(use_prob==0){
          train_examps[idx].update_weight=1;
        }
    }
    return;
  }
  
  }

    void Construct_IS_Distribution_Distributed(void) {
    int len=svm::train_examps.size;
    double prob_total,prob_tmp;    
    int start,end,idx;
    for(int i=0;i<nthreads;i++){
        start=i*(len/nthreads);
        end=start+(len/nthreads)+(len%nthreads)*(i==(nthreads-1));
        prob_total=0.0;
        prob_tmp=0.0;
        for(int j=start;j<end;j++){
            idx=Lip_sort[j];
            prob_total+=svm::train_examps[idx].probability;
        }
        for(int j=start;j<end;j++){
            idx=Lip_sort[j];
            svm::train_examps[idx].p_s=prob_tmp;
            svm::train_examps[idx].p_e=svm::train_examps[idx].p_s+svm::train_examps[idx].probability/prob_total;
            svm::train_examps[idx].update_weight=(svm::train_examps[idx].probability/prob_total)*(end-start);
            prob_tmp+=(svm::train_examps[idx].probability/prob_total);
            if(use_prob==0){
              svm::train_examps[idx].update_weight=1;
            }
        }
    }
    return;
  }
  

    void Generate_Sample_Sequence_Shared(int epoch) {
          int len=svm::train_examps.size/nthreads;  
          srand (epoch);
          int *seq;
          double rand_num;
          int sel_idx=0;
          int data_idx,left,right;
          bool found;
          int idx=0;
          for(int i=0;i<nthreads;i++){
              int start =i*len;
              seq = IS_seq[i];
              int trunk_len=len+(i==(nthreads-1))*(svm::train_examps.size%nthreads);
              for(int count=0;count<trunk_len;count++){
                  if(use_prob){
                      rand_num=(double)rand()/(RAND_MAX);
                      left=0,right=svm::train_examps.size;
                      found = 0;
                      while(left<=right){
                          data_idx=(left+right)/2;
                          sel_idx=data_idx;
                          if((rand_num>svm::train_examps[sel_idx].p_s)&&(rand_num<=svm::train_examps[sel_idx].p_e)){
                              found = 1;
                              break;
                          } else if(rand_num>svm::train_examps[sel_idx].p_e){
                              left = data_idx+1;
                          } else {
                              right = data_idx-1;
                          }
                      }
                      seq[count]=sel_idx;
                  } else {
                      seq[count]=rand()%trunk_len+start;
                  }
                  idx++;
              }
          }
          return;
      }
  
    void Generate_Sample_Sequence_Distributed(int epoch) {
          int len=svm::train_examps.size/nthreads;  
          srand (epoch);
          int *seq;
          double rand_num;
          int sel_idx;
          int data_idx,left,right;
          bool found;
          for(int i=0;i<nthreads;i++){
              int start =i*len;
              seq = IS_seq[i];
              int trunk_len=len+(i==(nthreads-1))*(svm::train_examps.size%nthreads);
              for(int count=0;count<trunk_len;count++){
                  if(use_prob){
                      rand_num=(double)rand()/(RAND_MAX);
                      left=start,right=start+trunk_len;
                      found = 0;
                      while(left<=right){
                          data_idx=(left+right)/2;
                          sel_idx=Lip_sort[data_idx];
                          if((rand_num>svm::train_examps[sel_idx].p_s)&&(rand_num<=svm::train_examps[sel_idx].p_e)){
                              found = 1;
                              break;
                          } else if(rand_num>svm::train_examps[sel_idx].p_e){
                              left = data_idx+1;
                          } else {
                              right = data_idx-1;
                          }
                      }
                      if(found==0)
                        continue;
                      seq[count]=sel_idx;
                  } else {
                      seq[count]=rand()%trunk_len+start;
                  }
              }
          }
          return;
      }
	
  template <class Model, class Params, class Exec>
  template <class Scan>
  int Hogwild<Model, Params, Exec>::UpdateModel(Scan &scan, int epoch) {
      scan.Reset();
      static int gen=0;
      
      if(gen==0){
          svm::Generate_Thread_Dataset();
          if((distributed)&&(use_prob)){
              //std::cout<<"Distributed Distribution Gen start\n";
              Construct_IS_Distribution_Distributed();
              //std::cout<<"Distributed Distribution Gen finished\n";
          } else {
              svm::Construct_IS_Distribution();
          }
          gen=1;
      }
      if(distributed){
          Generate_Sample_Sequence_Distributed(epoch);
      } else {
          Generate_Sample_Sequence_Shared(epoch);
      }
	  auto started_update = std::chrono::high_resolution_clock::now();
      Zero();
      if(svrg){
          //for gradient variance evaluation, we have to calculate optimal gradient
          model_.SnapShot();
      }
      FFAScan(model_, params_, scan, tpool_, Exec::UpdateModel, res_); 
      auto done_update = std::chrono::high_resolution_clock::now();
      return std::chrono::duration_cast<std::chrono::milliseconds>(done_update-started_update).count();
  }
  
  template <class Model, class Params, class Exec>
  template <class Scan>
  double Hogwild<Model, Params, Exec>::ComputeRMSE(Scan &scan) {
    scan.Reset();
    Zero();
    test_time_.Start();
    size_t count = FFAScan(model_, params_, scan,tpool_, Exec::TestModelRMSE, res_);
    test_time_.Stop();
  
    double sum_sqerr = 0;
    for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
      sum_sqerr += res_.values[i];
    }
	return sum_sqerr/count;
  }
  
  template <class Model, class Params, class Exec>
  template <class Scan>
  double Hogwild<Model, Params, Exec>::ComputeErrorRate(Scan &scan) {
    scan.Reset();
    Zero();
    test_time_.Start();
    size_t count = FFAScan(model_, params_, scan,tpool_, Exec::TestModelError, res_);
    test_time_.Stop();
    double err_count = 0;
    for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
      err_count += res_.values[i];
    }
    return (err_count/count);
  }  
  
  template <class Model, class Params, class Exec>
  template <class TrainScan>
  void Hogwild<Model, Params, Exec>::RunExperiment(int nepochs, TrainScan &trscan) {
    std::ofstream rmse_output;
    std::string fname="_rmse_";
    std::string xx="_lr_";
	std::string dir="./";
	std::string IS_RND_DIS="_IS_random_Dis";
	std::string IS_SHF_DIS="_IS_balanced_Dis";
	std::string SVRG="_SVRG";
    std::string fname1(szExampleFile);
    std::stringstream ini_lr;
	std::stringstream ld_str;
    ini_lr << std::fixed<<std::setprecision(10) << params_.step_size;
	ld_str << std::fixed<<std::setprecision(10) << params_.step_decay;
    std::string ini_lr_x(ini_lr.str());
    ini_lr_x.erase (ini_lr_x.find_last_not_of('0') + 1, std::string::npos );
	
    std::string ld_x(ld_str.str());
    ld_x.erase (ld_x.find_last_not_of('0') + 1, std::string::npos );	
	
    const size_t last_slash_idx = fname1.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        fname1.erase(0, last_slash_idx + 1);
    }
    double train_rmse,test_rmse,gn=0.0;
	std::string fn=dir+fname1;
	if(svrg){
		fn+=SVRG;
	} else {
		if(lipschitz){
			if(random_dis){
				fn+=IS_RND_DIS;
			}else{
				fn+=IS_SHF_DIS;
			}
		}
	}
	fn=fn+fname+std::to_string(tpool_.ThreadCount())+xx+ini_lr_x+"_ld_"+ld_x;
    if(svrg)
        std::cout<<"SVRG\n";
    std::cout<<"save to "<<fn<<std::endl;
    rmse_output.open(fn,std::ios_base::out);
    // Compute initial error state
    train_rmse = ComputeRMSE(trscan); 
    test_rmse = ComputeErrorRate(trscan);
    int time_avg_grad,real_time=0;    
    std::cout<<"sparsity:"<<check_sparsity(&model_)<<std::endl;
	auto abs_time_started = std::chrono::high_resolution_clock::now();
    for (int e = 1; e <= nepochs; e++) {
      eidx=e;
      auto started = std::chrono::high_resolution_clock::now();
      gn=Calculate_Full_Gradient_Parallel(svm::train_examps, params_, &model_);
      auto done = std::chrono::high_resolution_clock::now();
      time_avg_grad=std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();     
      real_time+=UpdateModel(trscan,e);
      if(svrg){
	  	//This is a hard request for SVRG, add to its time cost.
        real_time+=time_avg_grad;
      }
      train_rmse = ComputeRMSE(trscan);
      test_rmse = ComputeErrorRate(trscan);
      Exec::PostEpoch(model_, params_);
      std::cout<<"ep: "<<e<<" rmse: "<<train_rmse<<" error-rate: "<<test_rmse<<" grad_norm: "<<gn<<" time: "<<real_time<<std::endl;
      fflush(stdout);
      rmse_output<<std::setprecision(16)<<train_rmse<<" "<<test_rmse<<" "<<gn<<" "<<real_time<<std::endl;
    }
	auto abs_time_done = std::chrono::high_resolution_clock::now();
    std::cout<<"sparsity:"<<check_sparsity(&model_)<<std::endl;
	std::cout<<"absolute time "<<std::chrono::duration_cast<std::chrono::milliseconds>(abs_time_done-abs_time_started).count()<<std::endl;
    rmse_output.close();
  }
  
  } // namespace hogwild
  } // namespace haz


#endif


