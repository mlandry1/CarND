#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {

  x_ = x_in;  // object state
  P_ = P_in;  // object covariance matrix
  F_ = F_in;  // state transition matrix
  H_ = H_in;  // measurement matrix
  R_ = R_in;  // measurement covariance matrix
  Q_ = Q_in;  // process covariance matrix

}

void KalmanFilter::Predict() {
  /**
  TODO: Done!
    * predict the state
  */

  // states estimate update
  x_ = F_ * x_;

  // convariance update
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO: Done!
    * update the state by using Kalman Filter equations
  */

  // KF Measurement update step
  // error matrix (2D)
  VectorXd y_ = z - H_ * x_;

  // update common KF step
  UpdateKFstep(y_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO: Done!
    * update the state by using Extended Kalman Filter equations
  */
//  const float pi = 3.14159265358979323846;

  // recover state parameters
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // pre-compute a set of terms to avoid repeated calculation
  float rho = sqrt(px*px+py*py);

  // pre-compute phi
  float phi = atan2(py,px);

//  // normalize phi (between -pi and pi)
//  if(phi > pi){
//    phi = phi - pi;
//  }
//  else if(phi < -pi) {
//    phi = phi + pi;
//  }

  // pre-compute rho_dot
  float rho_dot = (px*vx + py*vy)/rho;

  // create hx non-linear vector
  VectorXd hx_(3);
  hx_<< rho, phi, rho_dot;

  // error matrix
  VectorXd y_ = z - hx_;

  // update common KF step
  UpdateKFstep(y_);
}

void KalmanFilter::UpdateKFstep(const VectorXd &y){
  // system uncertainty projection
  MatrixXd S_ = H_ * P_ * H_.transpose() + R_;

  // Kalman gain
  VectorXd K_ = P_ * H_.transpose() * S_.inverse();

  // states estimate update
  x_ = x_ + (K_ * y);
  long x_size = x_.size();

  // Uncertainty update
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K_ * H_) * P_;
}
