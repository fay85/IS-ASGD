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
#include <iostream>
#include <fstream>
#include <cmath>  
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#include "hazy/hogwild/tools-inl.h"
extern int use_log;
extern volatile int eidx;
namespace hazy {
namespace hogwild {
namespace svm {
	
	double inline sigmoid(double x){
		return 1.0/(1.0 + exp(x));
	}

	fp_type inline ComputeLoss(const SVMExample &e, const SVMModel& model) {
	  // determine how far off our model is for this example
	  vector::FVector<fp_type> const &w = model.weights;
	  fp_type dot = vector::Dot(w, e.vector);
	  double loss = (dot-e.value)*(dot-e.value);
	  return loss;
	}

	fp_type inline ComputeLoss_Log(const SVMExample &e, const SVMModel& model) {
	  // determine how far off our model is for this example
	  vector::FVector<fp_type> const &w = model.weights;
	  int label = e.value;
	  fp_type wxy = vector::Dot(w, e.vector);
	  wxy = wxy * e.value;
	  double loss = log(1+exp(-wxy));
	  return loss;
	}

	fp_type inline Error(const SVMExample &e, const SVMModel& model) {
	  // determine how far off our model is for this example
	  vector::FVector<fp_type> const &w = model.weights;
	  fp_type dot = vector::Dot(w, e.vector);
	  double predicted,error=0;
	  double loss = std::max((1 - dot * e.value), static_cast<fp_type>(0.0));
	  if((dot)>0){
		predicted=1.0;
	  }else{
		predicted=-1.0;
	  }
	  if(predicted!=e.value){
		error=1;
	  }
	  return error;
	}

	fp_type inline Error_Log(const SVMExample &e, const SVMModel& model) {
	  // determine how far off our model is for this example
	  vector::FVector<fp_type> const &w = model.weights;
	  fp_type wx = vector::Dot(w, e.vector);
	  double predicted=1/(1+exp(-wx)),error=0;
	  
	  if(predicted>0.5){
		predicted=1.0;
	  }else{
		predicted=-1.0;
	  }
	  if(predicted!=e.value){
		error=1;
	  }
	  return error;
	}

	void inline ModelUpdateSVRG(const SVMExample &examp, const SVMParams &params, SVMModel *model,vector::SVector<fp_type> *tmp) {
		vector::FVector<fp_type> &w = model->weights;
		vector::FVector<fp_type> &ss = model->weights_snapshot;
		vector::SVector<fp_type> &avgG = model->packed_true_gradient;
		//-------- calculate gardient of snapshot
		fp_type wxys = vector::Dot(ss, examp.vector);
		wxys = wxys * examp.value;
		tmp->size =0;
		if (wxys < 1) {
		  fp_type e = -2*(1-wxys)*params.step_size*examp.value;
		  for (size_t i = 0; i< examp.vector.size;i++ ) {
			tmp->values[i] = examp.vector.values[i] * e;
			tmp->index[i] =  examp.vector.index[i];
		  }
		  tmp->size = examp.vector.size;
		}
		// evaluate this example
		fp_type wxy = vector::Dot(w, examp.vector);
		wxy = wxy * examp.value;
		if (wxy < 1) { // hinge is active.
		fp_type e = 2*(1-wxy)*params.step_size * examp.value;
		for (size_t i = 0; i< examp.vector.size;i++ ) {
		  tmp->values[i] += examp.vector.values[i] * e;
		}
		}
		vector::ScaleAndAdd(w, *tmp, 1.0);	
		vector::ScaleAndAdd(w, avgG, -1.0*params.step_size);
		//L1-regularization
		fp_type * const vals = w.values;
		size_t const size = examp.vector.size;
		unsigned const * const degs = params.degrees;
		// update based on the evaluation
		fp_type const scalar = (params.step_size * params.mu);
		for (int i = size; i-- > 0; ) {
		  int const j = examp.vector.index[i];
		  unsigned const deg = degs[j];
		  vals[j] -= scalar / deg;
		}
	}

	void inline ModelUpdateSVRG_Log(const SVMExample &examp, const SVMParams &params, SVMModel *model,vector::SVector<fp_type> *tmp) {
		vector::FVector<fp_type> &w = model->weights;
		vector::FVector<fp_type> &ss = model->weights_snapshot;
		vector::SVector<fp_type> &avgG = model->packed_true_gradient;
		fp_type wxy = vector::Dot(ss, examp.vector);
		wxy = wxy * examp.value;
		double grad;
		grad = sigmoid(wxy)*params.step_size;
		grad*= examp.value;	  
		grad*=(-1);
		for (size_t i = 0; i< examp.vector.size;i++ ) {
		  tmp->values[i] = examp.vector.values[i] * grad;
		  tmp->index[i] =  examp.vector.index[i];
		}
		tmp->size = examp.vector.size;
		//-------------------------------
		// evaluate this example
		fp_type wxy_this = vector::Dot(w, examp.vector);
		wxy_this = wxy_this * examp.value;
		double grad_this;
		grad_this = sigmoid(wxy_this)*params.step_size;
		grad_this*= examp.value;
		for (size_t i = 0; i< examp.vector.size;i++ ) {
		  tmp->values[i] = examp.vector.values[i] * grad_this;
		  tmp->index[i] =  examp.vector.index[i];
		}
		vector::ScaleAndAdd(w, *tmp, 1.0);	
		vector::ScaleAndAdd(w, avgG, 1.0);	
		// L1-Regularization
		fp_type * const vals = w.values;
		size_t const size = examp.vector.size;
		unsigned const * const degs = params.degrees;
		// update based on the evaluation
		fp_type const scalar = (params.step_size * params.mu);
		for (int i = size; i-- > 0; ) {
			int const j = examp.vector.index[i];
			unsigned const deg = degs[j];
			vals[j] -= scalar / deg;
		}
	}

	void inline ModelUpdate_Log(const SVMExample &examp, const SVMParams &params, SVMModel *model) {
	  	vector::FVector<fp_type> &w = model->weights;
		fp_type wxy = vector::Dot(w, examp.vector);
		double grad;
		wxy = wxy * examp.value;
		grad = sigmoid(wxy)*params.step_size/examp.update_weight;
		grad*= examp.value;
	    vector::ScaleAndAdd(w, examp.vector, grad);  
		//L1-regularization
	    fp_type * const vals = w.values;
	    unsigned const * const degs = params.degrees;
	    size_t const size = examp.vector.size;
	    fp_type const scalar = params.step_size * params.mu/examp.update_weight;
	    for (int i = size; i-- > 0; ) {
	      int const j = examp.vector.index[i];
	      unsigned const deg = degs[j];
	      vals[j] -= scalar / deg;
	    }
	}

	void inline ModelUpdate(const SVMExample &examp, const SVMParams &params, SVMModel *model) {
	  vector::FVector<fp_type> &w = model->weights;
	  // evaluate this example
	  fp_type wx = vector::Dot(w, examp.vector);
	  fp_type wxy = wx * examp.value;

	    fp_type const e = 2*(examp.value-wx)*(params.step_size)/examp.update_weight;
	    vector::ScaleAndAdd(w, examp.vector, e);	  
		// L1-Regularization
	    fp_type * const vals = w.values;
	    unsigned const * const degs = params.degrees;
	    size_t const size = examp.vector.size;
	    fp_type const scalar = params.step_size * params.mu/examp.update_weight;
	    for (int i = size; i-- > 0; ) {
	      int const j = examp.vector.index[i];
	      unsigned const deg = degs[j];
	      vals[j] -= scalar / deg;
	    }
	}

	void SVMExec::PostUpdate(SVMModel &model, SVMParams &params) {
	  // Reduce the step size to encourage convergence
	  params.step_size *= params.step_decay;
	}

	void SVMExec::PostEpoch(SVMModel &model, SVMParams &params) {
	  // Reduce the step size to encourage convergence
	  params.step_size *= params.step_decay;
	}

	double SVMExec::UpdateModel(SVMTask &task, unsigned tid, unsigned total) {
	  SVMModel  &model = *task.model;
	  SVMParams const &params = *task.params;
	  vector::FVector<SVMExample> const & exampsvec = task.block->ex;
	  // calculate which chunk of examples we work on
	  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total); 
	  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);
	  // optimize for const pointers 
	  size_t *perm = task.block->perm.values;
	  SVMExample const * const examps = exampsvec.values;
	  SVMModel * const m = &model;
	  int idx;
	  double rand_num;
	  int *seq;
	  seq=IS_seq[tid];

	  vector::SVector<fp_type> tmp; 
	  if(svrg){
	    tmp.size = 0;
	    tmp.values = new fp_type[m->weights.size];
		tmp.index = new int[m->weights.size];
	  }

	  //std::cout<<"tid "<<tid<<" update model with length "<<end-start<<std::endl;
	  for (unsigned i = 0; i < end-start; i++) {
	    if((svrg==0)||(eidx==1)){
			if(use_log==0){
		      	ModelUpdate(examps[seq[i]], params, m);
			  } else {
			    ModelUpdate_Log(examps[seq[i]], params, m);
			  }
	    }else{
	      	if(use_log==0){
	      		ModelUpdateSVRG(examps[seq[i]], params, m, &tmp);   
	      	} else {
				ModelUpdateSVRG_Log(examps[seq[i]], params, m,&tmp); 
	      	}
	    }
	  }
	  if(svrg){
	    delete tmp.values;
		delete tmp.index;
	  }
	  //std::cout<<"update model fin\n";
	  return 0.0;
	}

	double SVMExec::TestModelRMSE(SVMTask &task, unsigned tid, unsigned total) {
	  SVMModel const &model = *task.model;
	  vector::FVector<SVMExample> const & exampsvec = task.block->ex;
	  // calculate which chunk of examples we work on
	  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total); 
	  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);
	  // keep const correctness
	  SVMExample const * const examps = exampsvec.values;
	  fp_type loss = 0.0,l;
	  // compute the loss for each example
	  for (unsigned i = start; i < end; i++) {
	  	if(use_log==0){
	    	l = ComputeLoss(examps[i], model);
	  	} else {
			l = ComputeLoss_Log(examps[i], model);
	    }
	    loss += l;
	  }
	  //std::cout<<"loss "<<loss<<std::endl;
	  return loss;
	}

	double SVMExec::TestModelError(SVMTask &task, unsigned tid, unsigned total) {
	  SVMModel const &model = *task.model;
	  vector::FVector<SVMExample> const & exampsvec = task.block->ex;
	  // calculate which chunk of examples we work on
	  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total); 
	  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);
	  // keep const correctness
	  SVMExample const * const examps = exampsvec.values;
	  fp_type error = 0.0,e;
	  // compute error for each example
	  for (unsigned i = start; i < end; i++) {
	  	if(use_log==0){
			e = Error(examps[i], model);
	  	} else {
			e = Error_Log(examps[i], model);
	  	}
	    error += e;
	  }
	  return error;
	}
  } // namespace svm
} // namespace hogwild

} // namespace hazy
