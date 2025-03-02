#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {

    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();
    //initialize the weights vector to 0
    for (size_t i = 0; i < 3; i++)
    {
        //The fifth one is for the bias term
        weights.push_back({ 0,0,0,0,0 });
    }

    weights[0] = computeGradientDescent(X_train, convertToBinaryModel(y_train, 1), weights[0]);
    weights[1] = computeGradientDescent(X_train, convertToBinaryModel(y_train, 2), weights[1]);
    weights[2] = computeGradientDescent(X_train, convertToBinaryModel(y_train, 3), weights[2]);

	/* Implement the following:
       	--- Initialize weights for each class
    	--- Loop over each class label
    	--- Convert the problem into a binary classification problem
        --- Loop over training epochs
       	--- Add bias term to the training example
    	--- Calculate weighted sum of features
        --- Calculate the sigmoid of the weighted sum
        --- Update weights using gradient descent
    */

    
    // TODO
}

std::vector<double> LogisticRegression::convertToBinaryModel(std::vector<double> y, double label) {
    std::vector<double> newClasses;
    for (size_t i = 0; i < y.size(); i++)
    {
        if (y[i] == label)
            newClasses.push_back(1);
        else
            newClasses.push_back(0);
    }
    return newClasses;
}

std::vector<double> LogisticRegression::computeGradientDescent(std::vector<std::vector<double>> X, std::vector<double> y, std::vector<double> weight) {
        
    std::vector<double> weights= weight;
    for (int i = 0; i < num_epochs; i++)
    {
        for (int j = 0; j < X.size(); j++)
        {
            double sigmoidValue= LogisticRegression::calcolateSigmoidValue(weights, X[j]);

            //<= is to condier also the bias term
            for (size_t x = 0; x <= X[0].size(); x++)
            {
                double gradientDescent;
                if (x == X[0].size()) {
                    gradientDescent = (sigmoidValue - y[j]);
                }
                else {
                    gradientDescent = (sigmoidValue - y[j]) * X[j][x];
                }
                weights[x] = weights[x] - (learning_rate * gradientDescent);
            }
        }
    }
    return weights;
}

double LogisticRegression::calcolateSigmoidValue(std::vector<double> weights,std::vector<double> x) {
    double weighetdSum = 0;
    for (size_t i = 0; i < x.size(); i++)
    {
        weighetdSum += x[i] * weights[i];
    }
    //add bias term
    weighetdSum += weights[x.size()];
    return 1 / (1 + exp(-1 * weighetdSum));
}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;
    for (int i = 0; i < X_test.size(); i++)
    {
        std::vector<double> probability;
        for (int j = 0; j < 3; j++)
        {
            probability.push_back(LogisticRegression::calcolateSigmoidValue(weights[j], X_test[i]));
        }
        double maxValue = probability[0];
        int prediction = 1;
        for (int j = 1; j < probability.size(); j++) {
            if (probability[j] > maxValue) {
                maxValue = probability[j];
                prediction = j+1;
            }
        }
        predictions.push_back(prediction);
    }


    /* Implement the following:
    	--- Loop over each test example
        --- Add bias term to the test example
        --- Calculate scores for each class by computing the weighted sum of features
        --- Predict class label with the highest score
    */
      
    
    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

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