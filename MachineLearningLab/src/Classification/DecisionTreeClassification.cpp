#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	for (std::vector<double> entry : X)
	{
		predictions.push_back(traverseTree(entry, root));
	}
	// Implement the function
	// TODO

	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

	/*
	Loop through candidate features and potential split thresholds:
You'll have to go through each feature and for each feature, consider various potential thresholds.
This involves iterating over the features and for each feature, iterating over the data points to find the best threshold.

Calculate Information Gain:
For each combination of feature and threshold, you'll need to calculate the information gain.
Information gain is a measure of the reduction in uncertainty or entropy achieved by making a particular split. Higher information gain indicates a better split.

Select the Best Split:
Keep track of the best information gain you've seen so far and the corresponding feature and threshold.
Update these values if you find a split with higher information gain.

Set the Split Index and Threshold:
After iterating through all features and potential thresholds, you will have the best split index and threshold.
	*/

	/* Implement the following:
		--- define stopping criteria
		--- Loop through candidate features and potential split thresholds.
		--- greedily select the best split according to information gain
		---grow the children that result from the split
	*/


	//stopping cretiria
	if (X.size() <= min_samples_split||depth>= max_depth) {
		return new Node(0, 0, nullptr, nullptr, mostCommonlLabel(y));
	}

	double best_gain = -1.0; // set the best gain to -1
	int split_idx = -1; // split index
	double split_thresh = -1; // split threshold

	//for each candidate I iterate first the row and than the column
	for (size_t i = 0; i < X[0].size(); i++)
	{
		std::vector<std::pair<double, double>> featuresTargets;
		std::vector<double> threshold;
		std::vector<double> column;
		std::vector<double> target;
		//create column vector
		for (size_t j = 0; j < X.size(); j++) {
			featuresTargets.push_back({ X[j][i], y[j]});
		}
		//sort the vector
		std::sort(featuresTargets.begin(), featuresTargets.end());
		for (std::pair<double, double> pair : featuresTargets) {
			column.push_back(pair.first);
			target.push_back(pair.second);
		}
		//compute the possible threshold values
		for (size_t j = 0; j < featuresTargets.size() - 1; j++) {
			threshold.push_back((column[j]));
		}
		//for each of the threshold value, verify if is the best
		for (size_t j = 0; j < threshold.size(); j++)
		{
			double gain = informationGain(target, column, threshold[j]);
			if (gain > best_gain) {
				best_gain = gain;
				//storing index feature with best information gain
				split_idx = i;
				split_thresh = threshold[j];
			}
		}
	}

	if (best_gain == 0.0) {
		return new Node(0, 0, nullptr, nullptr, mostCommonlLabel(y));
	}


	//split the tree
	std::vector<std::vector<double>> leftX, rightX;
	std::vector<double> leftY, rightY;
	for (size_t i = 0; i < X.size(); i++)
	{
		if (X[i][split_idx] > split_thresh) {
			rightX.push_back(X[i]);
			rightY.push_back(y[i]);
		}
		else {
			leftX.push_back(X[i]);
			leftY.push_back(y[i]);
		}
	}

	Node* left = growTree(leftX, leftY, depth + 1); // grow the left tree
	Node* right = growTree(rightX, rightY, depth + 1);  // grow the right tree
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);
	std::vector<int> leftIndexes;
	std::vector<int> rightIndexes;
	for (size_t i = 0; i < X_column.size(); i++)
	{
		if (X_column[i] > split_thresh) {
			rightIndexes.push_back(i);
		}
		else {
			leftIndexes.push_back(i);
		}
	}
	double leftEntropy = static_cast<double>(leftIndexes.size()) / X_column.size()*EntropyFunctions::entropy(y, leftIndexes);
	double rightEntropy= static_cast<double>(rightIndexes.size()) / X_column.size()* EntropyFunctions::entropy(y, rightIndexes);
	double ig = parent_entropy - (leftEntropy + rightEntropy);

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/


	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {
	std::unordered_map<double, int> frequencyMap;

	for (double label : y) {
		frequencyMap[label]++;
	}

	double most_common = y[0];
	int maxFrequency = 0;

	for (std::pair<double, int> entry : frequencyMap) {
		if (entry.second > maxFrequency) {
			maxFrequency = entry.second;
			most_common = entry.first;
		}
	}
	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

	if (node->isLeafNode()) {
		return node->value;
	}

	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	else {
		return traverseTree(x, node->right);
	}
	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
	DataPreprocessor DataPreprocessor;
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}