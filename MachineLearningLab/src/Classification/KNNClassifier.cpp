#include "KNNClassifier.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


											///  KNNClassifier class implementation  ///


// KNNClassifier function: Constructor for KNNClassifier class. //
KNNClassifier::KNNClassifier(int k) : k_(k) {}


// fit function: Fits the KNNClassifier with the given training data.//
void KNNClassifier::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
	X_train_ = X_train;
	y_train_ = y_train;
}


// predict function: Predicts the labels of a given set of test data points using the K-Nearest Neighbors algorithm.//
std::vector<double> KNNClassifier::predict(const std::vector<std::vector<double>>& X_test) const {
	std::vector<double> y_pred; // Store predicted labels for all test data points
	y_pred.reserve(X_test.size()); // Reserve memory for y_pred to avoid frequent reallocation

	// Check if training data is empty
	if (X_train_.empty() || y_train_.empty()) {
		throw std::runtime_error("Error: Empty training data.");
	}

    // Loop through each test data point
    for (size_t i = 0; i < X_test.size(); i++)
    {
        std::vector<std::pair<double, int>> distances; // Declare a new vector for storing the distances
        for (size_t j = 0; j < X_train_.size(); j++)
        {
            // Calculate Euclidean distance between test data point and each training data point
            distances.push_back({ SimilarityFunctions::euclideanDistance(X_test[i], X_train_[j]), j });
            // Calculate Manhattan distance between test data point and each training data point
            //distances.push_back({ SimilarityFunctions::manhattanDistance(X_test[i], X_train_[j]), j});
        }

        // Sort the array 
        std::sort(distances.begin(), distances.end());

        // Store the first k-elements in a new array
        std::vector<int> k_indexes;
        for (size_t j = 0; j < k_; j++)
        {
            k_indexes.push_back(distances[j].second);
        }


        // Retrieve the label of the first k-elements
        std::vector<double> labels;
        for (size_t j = 0; j < k_; j++)
        {
            labels.push_back(y_train_[k_indexes[j]]);
        }
        
        // Compute the most common label
        std::vector<double> counter = {0, 0, 0};
        for (size_t j = 0; j < k_; j++)
        {
            if (labels[j] == 1)
            {
                counter[0] += 1;
            }
            if (labels[j] == 2)
            {
                counter[1] += 1;
            }
            if (labels[j] == 3)
            {
                counter[2] += 1;
            }
        }


        // Storing first array element in "y_pred" variable
        double max = counter[0];
        double label = 1;
        for (int i = 1; i < 3; i++) {
            if (max < counter[i]) {
                max = counter[i];
                label = i + 1;
            }
        }

        y_pred.push_back(label);
    }

	
	return y_pred; // Return vector of predicted labels for all test data points
}



/// runKNNClassifier: this function runs the KNN classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
    KNNClassifier::runKNNClassifier(const std::string& filePath, int trainingRatio) {
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
        fit(trainData, trainLabels);
        
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


