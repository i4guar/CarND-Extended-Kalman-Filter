#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  
  // measurements to state matrix - radar (approximated with jacobian)
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  
  // covariance matrix - state  
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;
  // measurements to state matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      ekf_.x_ = VectorXd(4);
      ekf_.x_ << measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]),
      measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]),
      0,
      0;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //Initialize state.
	  ekf_.x_ = VectorXd(4);
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
              measurement_pack.raw_measurements_[1], 
              0, 
              0;
    }

    
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  
  // delta t in seconds time since last timestamp
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  // Update previous timestamp
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // update state transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;
  
  
  // dt to different powers - squared, ...
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  
  // update pocess noise covariance matrix
  float noise_ax = 9;
  float noise_ay = 9;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
         0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
         dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
         0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  // predict
  ekf_.Predict();

  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    std::cout << "RADAR" << std::endl;
    // Radar updates
    
    // update measurement covariance matrix
	ekf_.R_ = R_radar_;
    
    // update measurement to state matrix
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    
    // extended kalman filter update
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    std::cout << "Laser" << std::endl;
    // Laser updates
    
    // update measurement covariance matrix
    ekf_.R_ = R_laser_;
    
    // update measurement to state matrix
    ekf_.H_ = H_laser_;
    
    // kalman filter update
	ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
