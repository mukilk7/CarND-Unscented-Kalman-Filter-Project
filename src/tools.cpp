#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() <= 0 || estimations.size() != ground_truth.size()) {
    return rmse;
  }

  //Compute sum of squared residuals
  for (int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse = rmse + residual;
  }
  //Average the sum
  rmse = rmse / estimations.size();
  //Compute square root of average sum
  rmse = rmse.array().sqrt();
  return rmse;
}

float Tools::NormalizeAngle(float a) {
  //deterministically running normalization logic
  if (fabs(a) > M_PI) {
    a = a - (round(a / (2 * M_PI)) * (2 * M_PI));
  }
  return a;
}