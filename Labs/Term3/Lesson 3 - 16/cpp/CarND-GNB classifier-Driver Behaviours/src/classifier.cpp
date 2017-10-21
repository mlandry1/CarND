#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

vector<double> GNB::compute_mean(vector< vector<double >> total){
  vector<double> mean;

  // mean init
  for(int i=0; i<4; i++)
    mean.push_back(0.0);

  for(int i=0; i<total.size(); i++){
    for(int j=0; j<total[i].size(); j++)
      mean[j] = mean[j] + total[i][j];
  }
  for(int j=0; j<total[0].size(); j++){
    mean[j] = mean[j]/total.size();
  }

  return mean;
}

vector<double> GNB::compute_std(vector< vector<double >> total, vector<double> mean){
  vector<double> std;

  // std vector init
  for(int i=0; i<4; i++)
    std.push_back(0.0);

  for(int i=0; i<total.size(); i++){
    for(int j=0; j<total[i].size(); j++)
      std[j] = std[j] + pow((total[i][j] - mean[j]), 2.0);
  }
  for(int j=0; j<total[0].size(); j++){
    std[j] = sqrt(std[j]/total.size());
  }

  return std;
}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
    // create a list of states vector for each label
    vector<vector< vector<double >>> total_by_label;
    vector< vector<double >> total_left;
    vector< vector<double >> total_keep;
    vector< vector<double >> total_right;

    for(int i=0; i< data.size(); i++){
      vector<double> states = data[i];
      string label = labels[i];

      // process the raw s,d,s_dot,d_dot snapshot if desired
      states = process_vars(states);

      if(label == possible_labels[0])
        total_left.push_back(states);

      if(label == possible_labels[1])
        total_keep.push_back(states);

      if(label == possible_labels[2])
        total_right.push_back(states);
    }

    total_by_label.push_back(total_left);
    total_by_label.push_back(total_keep);
    total_by_label.push_back(total_right);

    // compute the mean vector for each label
    vector< vector<double >> means;
    vector<double> mean;

    for(int i=0; i<total_by_label.size(); i++){
      mean = this->compute_mean(total_by_label[i]);
      means.push_back(mean);
    }

    // compute the std vector for each label
    vector< vector<double >> stds;
    vector<double> std;

    for(int i=0; i<total_by_label.size(); i++){
      std = this->compute_std(total_by_label[i], means[i]);
      stds.push_back(std);
    }

    this->_means = means;
    this->_stds = stds;
}

// helper functions
double GNB::gaussian_prob(double obs, double mu, double sig)
{
  double num = pow((obs - mu), 2.0);
  double denum = 2*pow(sig, 2.0);
  double norm = 1 / sqrt(2*M_PI*pow(sig, 2.0));

  return norm * exp(-num/denum);
}

vector<double> GNB::process_vars(vector<double> vars){
  return vars;
}

vector<double> GNB::_predict(vector<double> obs){
  vector<double> probs;

  obs = process_vars(obs);

  double likelihood;

  for(int i=0; i<possible_labels.size(); i++){
    double product = 1.0;
    for(int j=0; j<obs.size(); j++){
      likelihood = this->gaussian_prob(obs[j], _means[i][j], _stds[i][j]);
      product *= likelihood;
    }
    probs.push_back(product);
  }
  // sum of probablities
  double t = 0.0;
  for(int i=0; i<possible_labels.size(); i++){
    t += probs[i];
  }

  for(int i=0; i<possible_labels.size(); i++){
    probs[i] = probs[i]/t;
  }

  return probs;
}

string GNB::predict(vector<double> sample)
{
  vector<double> probs = this->_predict(sample);

  int idx = 0;
  double best_p = 0.0;

  for(int i=0; i<possible_labels.size(); i++){
    if(probs[i] > best_p){
      best_p = probs[i];
      idx = i;
    }
  }

	return this->possible_labels[idx];

}
