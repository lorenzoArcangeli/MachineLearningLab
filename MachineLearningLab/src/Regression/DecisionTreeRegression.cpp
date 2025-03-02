#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	for (std::vector<double> entry : X)
	{
		predictions.push_back(traverseTree(entry, root));
	}
	// Implement the function
	// TODO

	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

	//stopping criteria
	if (X.size() <= min_samples_split || depth >= max_depth) {
		return new Node(0, 0, nullptr, nullptr, mean(y));
	}

	int split_idx = -1;
	double split_thresh = 0.0;
	double bestMeanSquaredError = 1.7976931348623157e+308;


	//for each candidate I iterate first the row and than the column
	for (size_t i = 0; i < X[0].size(); i++)
	{
		std::vector<double> column;
		std::vector<double> threshold;
		//create column vector
		for (size_t j = 0; j < X.size(); j++) {
			column.push_back(X[j][i]);
		}
		//compute the possible threshold values
		for (size_t j = 0; j < column.size() - 1; j++) {
			threshold.push_back((column[j] + column[j + 1]) / 2);
		}
		//for each of the threshold value, verify if is the best
		for (size_t j = 0; j < threshold.size(); j++)
		{
			double mse = meanSquaredError(y, column, threshold[j]);
			if (mse <  bestMeanSquaredError) {
				bestMeanSquaredError = mse;
				//storing index feature with best information gain
				split_idx = i;
				split_thresh = threshold[j];
			}
		}
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
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- Find the best split threshold for the current feature.
		--- grow the children that result from the split
	*/
	
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	double mse = 0.0;
	std::vector<double> left;
	std::vector<double> right;
	for (size_t i = 0; i < X_column.size(); i++)
	{
		if (X_column[i] > split_thresh) {
			right.push_back(y[i]);
		}
		else {
			left.push_back(y[i]);
		}
	}
	double leftMean = mean(left);
	double rightMean = mean(right);
	for (double value : left) {
		mse += pow(value - leftMean, 2);
	}

	for (double value : right) {
		mse += pow(value - rightMean, 2);
	}
	// Calculate the mse
	
	return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;
	double sum = 0.0;
	
	for (size_t i = 0; i < values.size(); i++)
	{
		sum += values[i];
	}
	meanValue = sum / values.size();
	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

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
	// TODO

	return 0.0;
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

