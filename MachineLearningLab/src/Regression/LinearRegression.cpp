#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///


// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

	// This implementation is using Matrix Form method
	/* Implement the following:	  
	    --- Check if the sizes of trainData and trainLabels match
	    --- Convert trainData to matrix representation
	    --- Construct the design matrix X
		--- Convert trainLabels to matrix representation
		--- Calculate the coefficients using the least squares method
		--- Store the coefficients for future predictions
	*/

    if (trainData.size() == trainLabels.size()) {
        Eigen::MatrixXd features(trainData.size(), trainData[0].size());
        Eigen::VectorXd labels(trainLabels.size());
        for (size_t i = 0; i < trainData.size(); i++)
        {
            for (size_t j = 0; j < trainData[0].size(); j++)
            {
                features(i, j) = trainData[i][j];
            }
            labels(i) = trainLabels[i];
        }
        m_coefficients = (features.transpose() * features).inverse() * features.transpose() * labels;
    }
    
}


// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

	// This implementation is using Matrix Form method    
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
   // if (testData[0].size() == m_coefficients.size()) {
        Eigen::MatrixXd features(testData.size(), testData[0].size());
        for (size_t i = 0; i < testData.size(); i++)
        {
            for (size_t j = 0; j < testData[0].size(); j++)
            {
                features(i, j) = testData[i][j];
            }
        }
        /*for (size_t i = 0; i < testData.size(); i++)
        {
            predictedValues(i) = features * m_coefficients;
        }*/
        Eigen::VectorXd predictedValues = features * m_coefficients;
        std::vector<double> result(predictedValues.data(), predictedValues.data() + predictedValues.size());
        return result;
    
}
/*
void LinearRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train, bool gd) {

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

/*
    // TODO
}


std::vector<double> LinearRegression::convertToBinaryModel(std::vector<double> y, double label) {
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

std::vector<double> LinearRegression::computeGradientDescent(std::vector<std::vector<double>> X, std::vector<double> y, std::vector<double> weight) {

    std::vector<double> weights = weight;
    for (int i = 0; i < num_epochs; i++)
    {
        for (int j = 0; j < X.size(); j++)
        {
            double sigmoidValue = LinearRegression::calcolateSigmoidValue(weights, X[j]);

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

double LinearRegression::calcolateSigmoidValue(std::vector<double> weights, std::vector<double> x) {
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
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X_test, bool gd) {
    std::vector<double> predictions;
    for (int i = 0; i < X_test.size(); i++)
    {
        std::vector<double> probability;
        for (int j = 0; j < 3; j++)
        {
            probability.push_back(LinearRegression::calcolateSigmoidValue(weights[j], X_test[i]));
        }
        double maxValue = probability[0];
        int prediction = 1;
        for (int j = 1; j < probability.size(); j++) {
            if (probability[j] > maxValue) {
                maxValue = probability[j];
                prediction = j + 1;
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


  /*  return predictions;
}
*/
/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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