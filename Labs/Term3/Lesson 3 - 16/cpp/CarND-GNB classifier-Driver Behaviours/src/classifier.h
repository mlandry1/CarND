#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};

	vector< vector<double >> _means;
	vector< vector<double >> _stds;

	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	vector<double> process_vars(vector<double> vars);
 	double gaussian_prob(double obs, double mu, double sig);
 	vector<double> compute_mean(vector< vector<double >> total);
  vector<double> compute_std(vector< vector<double >> total, vector<double> mean);
  vector<double> _predict(vector<double> obs);

 	void train(vector<vector<double> > data, vector<string>  labels);

  string predict(vector<double>);

};

#endif
