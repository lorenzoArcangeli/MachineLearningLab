#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include <string>
#include <vector>
#include <utility>


/// LogisticRegression class definition ///



class LogisticRegression {
public:
    LogisticRegression(double learning_rate = 0.0001, int num_epochs = 10); 
    void fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train);
    std::vector<double> predict(const std::vector<std::vector<double>>& X_test);
    //add
    std::vector<double> computeGradientDescent(std::vector<std::vector<double>> X, std::vector<double> y, std::vector<double> weight);
    std::vector<double> convertToBinaryModel(std::vector<double> y, double label);
    double calcolateSigmoidValue(std::vector<double> weights, std::vector<double> x);

    std::tuple<double, double,
        std::vector<double>, std::vector<double>,
        std::vector<double>, std::vector<double>>
        runLogisticRegression(const std::string& filePath, int trainingRatio);

private:
    double learning_rate;
    int num_epochs;
    std::vector<std::vector<double>> weights;

    // Sigmoid function
    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }
};

#endif // LOGISTIC_REGRESSION_H


