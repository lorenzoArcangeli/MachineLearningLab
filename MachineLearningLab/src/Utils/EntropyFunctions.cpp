#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {

	// Convert labels to unique integers and count their occurrences
	//TODO

	// Compute the probability and entropy
	//TODO
	int total_samples = y.size();

	//idk perch� ma non lo uso ahahah
	std::vector<double> hist;

	std::unordered_map<double, double> label_map;
	double entropy = 0.0;

	for (double label : y) {
		label_map[label]++;
	}

	for (std::pair<double, double> entry : label_map) {
		double probability = entry.second / total_samples;
		entropy += -probability * log(probability);
	}
	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, double> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;
	for (size_t i = 0; i < total_samples; i++)
	{
		label_map[y[idxs[i]]]++;
	}

	for (std::pair<double, double> entry : label_map) {
		double probability = entry.second / total_samples;
		entropy += -probability * log(probability);
	}
	// Convert labels to unique integers and count their occurrences
	//TODO


	// Compute the probability and entropy
	//TODO


	return entropy;
}


