#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


UKF::UKF(Tools t): UKF() {
  tools = t;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* laser measuremnt dimension
  n_z_laser_ = 2;

  ///* radar measuremnt dimension
  n_z_radar_ = 3;

  n_sigpts_ = 2 * n_aug_ + 1;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // laser measurement model matrix
  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.25;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigpts_);

  ///* time when the state is true, in us
  time_us_ = 0;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  ///* Weights of sigma points
  weights_ = VectorXd(n_sigpts_);
  double weight = 0.5 / (n_aug_ + lambda_);
  weights_.fill(weight);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  R_laser_ = MatrixXd::Zero(n_z_laser_, n_z_laser_);
  R_laser_(0,0) = std_laspx_ * std_laspx_;
  R_laser_(1,1) = std_laspy_ * std_laspy_;

  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  R_radar_(0,0) = std_radr_ * std_radr_;
  R_radar_(1,1) = std_radphi_ * std_radphi_;
  R_radar_(2,2) = std_radrd_ * std_radrd_;

  ///* the current NIS for radar
  NIS_radar_ = 0;

  ///* the current NIS for laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rodot = meas_package.raw_measurements_[2];
      float px = ro * cos(phi);
      float py = ro * sin(phi);
      x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      //Get px, py directly from the raw_mesaurements
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;

  /*** PREDICTION **`*/

  tools.DebugLog("Calling state prediction...");
  Prediction(delta_t);

  /*** UPDATE ***/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    tools.DebugLog("Calling radar update...");
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    tools.DebugLog("Calling lidar update...");
    UpdateLidar(meas_package);
  }

  std::cout << "meas = " << meas_package.raw_measurements_ << std::endl;
  std::cout << "x_ = " << x_ << std::endl;
  std::cout << "P_ = " << P_ << std::endl;
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigpts_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_ * std_a_, 0,
        0, std_yawdd_ * std_yawdd_;
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2,2) = Q;
  
  //create square root matrix
  MatrixXd lproot = (lambda_ + n_aug_) * P_aug;
  lproot = lproot.llt().matrixL();
  
  //create augmented sigma points
  for (int i = 0; i < n_aug_; i++) {
      VectorXd row = VectorXd(n_sigpts_);
      //calculate sigma points ...
      VectorXd b2 = x_aug(i) + lproot.row(i).array();
      VectorXd b3 = x_aug(i) - lproot.row(i).array();
      row << x_aug(i), b2, b3;
      //set sigma points as columns of matrix Xsig
      Xsig_aug.row(i) = row;
  }

  return Xsig_aug;
}

void UKF::PredictSigmaPoints(const MatrixXd Xsig_aug, const double delta_t) {
  for (int i = 0; i < Xsig_aug.cols(); i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  //Generate Sigma Points
  MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints();

  //Predict Sigma Points
  PredictSigmaPoints(Xsig_aug, delta_t);

  //Predict state mean
  x_.fill(0);
  for (int i = 0; i < n_sigpts_; i++) {
      x_ += Xsig_pred_.col(i) * weights_(i);
  }
  
  //Predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sigpts_; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    x_diff(3) = tools.NormalizeAngle(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::GenericKalmanUpdate(VectorXd z, VectorXd z_pred, MatrixXd Zsig, MatrixXd S, MatrixXd Tc, double &NIS_out) {
  //calculate Kalman gain K;
  MatrixXd K = MatrixXd::Zero(n_x_, z.size());
  K = Tc * S.inverse();
  
  tools.DebugLog("Kalman gain calc Done");

  //update state mean
  VectorXd y = (z - z_pred);
  x_ = x_ + K * y;

  tools.DebugLog("Mean state update Done");

  //update covariance matrix
  P_ = P_ - K * S * K.transpose();

  tools.DebugLog("state covariance matrix update Done");

  //Calculate NIS
  NIS_out = y.transpose() * S.inverse() * y;

  tools.DebugLog("NIS_out calc Done");
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  * Use lidar data to update the belief about the object's
  * position. Modify the state vector, x_, and covariance, P_.
  *
  * Also calculate the lidar NIS.
  */

  tools.DebugLog("LIDAR> Starting lidar update");

  VectorXd z = meas_package.raw_measurements_;

  //transform prediction in ctrv space to radar measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_laser_, n_sigpts_);
  for (int i = 0; i < n_sigpts_; i++) {
    Zsig.col(i) << Xsig_pred_(0, i), Xsig_pred_(1, i);
  }
  tools.DebugLog("LIDAR> Prediction to Lidar Meas Conversion Done");

  //calculate mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z_laser_);
  for (int i = 0; i < n_sigpts_; i++) {
      z_pred = z_pred + Zsig.col(i) * weights_(i);
  }
  tools.DebugLog("LIDAR> Mean pred measurement calc Done");

  //calculate measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z_laser_,n_z_laser_);
  for (int i = 0; i < n_sigpts_; i++) {
      VectorXd zdiff = Zsig.col(i) - z_pred;
      S = S + weights_(i) * zdiff * zdiff.transpose();
  }
  S = S + R_laser_;

  tools.DebugLog("LIDAR> Measurement covariance matrix calc Done");

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
  for (int i = 0; i < n_sigpts_; i++) {
      VectorXd xdiff = Xsig_pred_.col(i) - x_;
      VectorXd zdiff = Zsig.col(i) - z_pred;
      Tc = Tc + weights_(i) * xdiff * zdiff.transpose();
  }

  tools.DebugLog("LIDAR> Cross correlation matrix calc Done");

  GenericKalmanUpdate(z, z_pred, Zsig, S, Tc, NIS_laser_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  * Use radar data to update the belief about the object's
  * position. Modify the state vector, x_, and covariance, P_.
  *
  * Also calculate the radar NIS.
  */

  tools.DebugLog("RADAR> Starting radar update");

  VectorXd z = meas_package.raw_measurements_;

  //transform prediction in ctrv space to radar measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sigpts_);
  for (int i = 0; i < n_sigpts_; i++) {
    VectorXd c2p = VectorXd(n_z_radar_);
    tools.CartesianToPolar(Xsig_pred_.col(i), c2p);
    Zsig.col(i) = c2p;
  }
  tools.DebugLog("RADAR> Prediction to Radar Meas Conversion Done");

  //calculate mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  for (int i = 0; i < n_sigpts_; i++) {
      z_pred = z_pred + Zsig.col(i) * weights_(i);
  }
  tools.DebugLog("RADAR> Mean pred measurement calc Done");

  //calculate measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z_radar_,n_z_radar_);
  for (int i = 0; i < n_sigpts_; i++) {
      VectorXd zdiff = Zsig.col(i) - z_pred;
      zdiff(1) = tools.NormalizeAngle(zdiff(1));
      S = S + weights_(i) * zdiff * zdiff.transpose();
  }
  S = S + R_radar_;

  tools.DebugLog("RADAR> Measurement covariance matrix calc Done");

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
  for (int i = 0; i < n_sigpts_; i++) {
      VectorXd xdiff = Xsig_pred_.col(i) - x_;
      xdiff(3) = tools.NormalizeAngle(xdiff(3));
      VectorXd zdiff = Zsig.col(i) - z_pred;
      zdiff(1) = tools.NormalizeAngle(zdiff(1));
      Tc = Tc + weights_(i) * xdiff * zdiff.transpose();
  }

  tools.DebugLog("RADAR> Cross correlation matrix calc Done");

  GenericKalmanUpdate(z, z_pred, Zsig, S, Tc, NIS_radar_);
}
