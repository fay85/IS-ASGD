
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

#include "hazy/vector/fvector.h"
#include "svmmodel.h"

namespace hazy {
namespace hogwild {
namespace svm {
    std::vector<fp_type*> d_release;
    std::vector<int*> i_release;
    
template <class Scan>
size_t LoadSVMExamples(Scan &scan, vector::FVector<SVMExample> &ex) {
  std::vector<SVMExample> examps;
  int lastrow = -1;
  double rating = 0.0;
  std::vector<fp_type> data;
  std::vector<int> index;

  int max_col = 0;

  while (scan.HasNext()) {
    const types::Entry &e = scan.Next();
    if (lastrow == -1) {
      lastrow = e.row;
    }
    if ((lastrow != e.row) || (!scan.HasNext())) {
      // finish off the previous vector and start a new one
      lastrow = e.row;
      fp_type *d = new fp_type[data.size()];
      int *i = new int[data.size()];
      for (size_t j = 0; j < data.size(); j++) {
        d[j] = data[j];
        i[j] = index[j];
      }
      SVMExample temp(rating, d, i, data.size(),1.0);
      examps.push_back(temp);
      rating = 0.0;
      data.clear();
      index.clear();
    }

    if (e.col < 0) {
      rating = e.rating;
    } else {
      if (e.col > max_col) {
        max_col = e.col;
      }
      data.push_back(e.rating);
      index.push_back(e.col);
    }
  }

  // Copy from temp vector into persistent memory
  ex.size = examps.size();
  ex.values = new SVMExample[ex.size];
  for (size_t i = 0; i < ex.size; i++) {
    new (&ex.values[i]) SVMExample(examps[i]);
    for (size_t j = 0; j < ex.values[i].vector.size; j++) {
      assert(ex.values[i].vector.index[j] >= 0);
      assert(ex.values[i].vector.index[j] <= max_col);
    }
  }
  return max_col+1;
}

size_t LoadLibSVMStyleFiles(std::string file, vector::FVector<SVMExample> &ex) {
  std::vector<SVMExample> examps;
  double rating = 0.0;
  std::vector<fp_type> data;
  std::vector<int> index;
  int max_col = 0;
  int current_count=0;
  std::ifstream fin;
  std::string line;
  int rating_range_1=0;
  int rating_range_0=0;
  int hinge = 0;
  fin.open(file);
  while (getline(fin, line)){
	  if(line.length()){
		  if(line[0] != '#' && line[0] != ' '){
		  	  //parse and load sample
			  std::vector<std::string> tokens = split(line,' ');
			  rating = atoi(tokens[0].c_str());
              //std::cout<<rating<<std::endl;
              if(rating==1){
                //std::cout<<"xx\n";
                rating_range_1++;
              }
              if(rating!=1){
                //std::cout<<"yy\n";
                rating_range_0++;
                rating = -1;
              }              

			  for(unsigned int i = 1; i < tokens.size(); i++){
					  std::vector<std::string> feat_val = split(tokens[i],':');
					  if(feat_val.size() == 2){
						  data.push_back(atof(feat_val[1].c_str()));
						  index.push_back(atoi(feat_val[0].c_str())-1);
						  if (atoi(feat_val[0].c_str()) > max_col) {
							max_col = atoi(feat_val[0].c_str());
						  }
					  }
			  }
			  //push sample to samples
		      fp_type *d = new fp_type[data.size()];
		      int *i = new int[data.size()];
              d_release.push_back(d);
              i_release.push_back(i);
		      for (size_t j = 0; j < data.size(); j++) {
		        d[j] = data[j];
		        i[j] = index[j];
				if(i[j]<0){
					std::cout<<"current_count "<<current_count<<" "<<i[j]<<std::endl;
					assert(0);
				}
		      }
		      SVMExample temp(rating, d, i, data.size(),1.0);
		      examps.push_back(temp);
		      data.clear();
		      index.clear();
			  //if(verbose) cout << "read example " << data.size() << " - found " << example.size()-1 << " features." << endl; 
			  current_count++;
		  }    
	  }
	  
  }
  fin.close();
  std::cout<<"1 "<<rating_range_1<<std::endl;
  std::cout<<"-1 "<<rating_range_0<<std::endl;
  assert(rating_range_1!=0);
  assert(rating_range_0!=0);
  // Copy from temp vector into persistent memory
  ex.size = examps.size();
  ex.values = new SVMExample[ex.size];
  for (size_t i = 0; i < ex.size; i++) {
    new (&ex.values[i]) SVMExample(examps[i]);
    for (size_t j = 0; j < ex.values[i].vector.size; j++) {
	  if(ex.values[i].vector.index[j] < 0){
	  std::cout<<"a "<<ex.values[i].vector.index[j]<<std::endl;
	  	assert(0);
	  }
      if(ex.values[i].vector.index[j]>max_col)
      {
      	std::cout<<"b "<<ex.values[i].vector.index[j]<<std::endl;
      	assert(0);
      }
    }
  }

  std::cout<<"Total loaded examples "<<examps.size()<<" with feature count "<<max_col<<std::endl;
  std::cout<<"+1 label "<<rating_range_1<<" -1 label "<<rating_range_0<<std::endl;
  return max_col+1;
}

void LoadProbability(std::string file, vector::FVector<SVMExample> &ex) {
  std::ifstream fin;
  std::string line;
  double prob=0.0;
  double prob_total=0.0;
  fin.open(file);
  int idx=0;
  std::cout<<"Loading prob file "<<file<<std::endl;
  
  while (getline(fin, line)){
	  if(line.length()){
      	  //parse and load sample
    	  std::vector<std::string> tokens = split(line,' ');
    	  prob = atof(tokens[0].c_str());
          //std::cout<<"lding prob "<<prob<<std::endl;
	      prob_total+=prob;
        ex[idx].probability=prob;
        idx++;
	  }
  }
  fin.close();
  for(int i=0;i<idx;i++){
    ex[i].probability/=prob_total;
  }
  assert(idx>0);
  return;
}

void LoadDataSeg(std::string file) {
  std::ifstream fin;
  std::string line;
  fin.open(file);
  int idx=0;
  std::cout<<"Loading norm sort file "<<file<<std::endl;
  
  while (getline(fin, line)){
    if(line.length()){
          //parse and load sample
        std::vector<std::string> tokens = split(line,' ');
        idx = atoi(tokens[0].c_str());
        Lip_sort.push_back(idx);
    }
  }
  fin.close();
//  for (std::vector<int>::iterator it = Lip_sort.begin() ; it != Lip_sort.end(); ++it)
//    std::cout << *it<<std::endl;
    assert(Lip_sort.size()!=0);
  return;
}

/*! \brief Computes the degree of each feature, assuems degs init'd to all 0
 */
void CountDegrees(const vector::FVector<SVMExample> &ex, unsigned *degs) {
  for (size_t i = 0; i < ex.size; i++) {
    for (size_t j = 0; j < ex.values[i].vector.size; j++) {
      degs[ex.values[i].vector.index[j]]++;
    }
  }
}

}
}
}
