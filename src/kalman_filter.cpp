#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;


/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict next state
  x_ = F_ * x_;
  // Calculate uncertainty
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

// Update Kalman filter with measurment z
void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  // dimension of state x
  long x_size = x_.size();
  // identity matrix of dimension x_size
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  // new uncertainty of current state
  P_ = (I - K * H_) * P_;
}


// Update Kalman filter using the extended Kalman filter method with measurment z
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  VectorXd h = MatrixXd(3, 1);
  float rho = sqrt(px*px + py*py);
  h << rho, 
  atan2(py,px),
  (px*vx + py * vy) / rho;
  
  VectorXd y = z - h;
  
  // ensure that phi of y is inside +/- PI
  y(1) = fmod(y(1), M_PI);
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  
  //new estimate
  x_ = x_ + (K * y);
  // dimension of state x
  long x_size = x_.size();
  // identity matrix of dimension x_size
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  // new uncertainty
  P_ = (I - K * H_) * P_;   
}
