#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::endl;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}


// Calculate the root mean squared error
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  // init root mean squared error vector
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }
  
  // calculate sum
  for (unsigned int i=0; i < estimations.size(); ++i) {

    // calculate residual
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // calculate mean
  rmse = rmse/estimations.size();

  // calculate squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

// Calculate Jacobian Matrix
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // init matrix 
  MatrixXd Hj(3,4);
  // recover state parameters
  // position x
  float px = x_state(0);
  // position y
  float py = x_state(1);
  // velocity x
  float vx = x_state(2);
  // velocity y
  float vy = x_state(3);
  
  // denominators
  float d1 = px*px + py*py;
  float d2 = sqrt(d1);
  float d3 = d1 * d2;

  // if denominator to small or zero calculation of jacobian won't work
  if(fabs(d1) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }
  
  // calculate jacobian
  Hj << (px/d2), (py/d2), 0, 0,
      -(py/d1), (px/d1), 0, 0,
      py*(vx*py - vy*px)/d3, px*(px*vy - py*vx)/d3, px/d2, py/d2;
  
  return Hj;
} 
