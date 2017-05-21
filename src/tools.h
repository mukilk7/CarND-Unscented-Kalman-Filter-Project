#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include <cstdarg>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  bool enableDebugLogging = false;

  /**
  * A helper method to calculate RMSE.
  */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

  /**
  * A helper method to normalize angles deterministically.
  */
  float NormalizeAngle(float input_angle);

  void CartesianToPolar(const Eigen::VectorXd &x, Eigen::VectorXd &out);

  void DebugLog(const char *sfmt, ...);

};

#endif /* TOOLS_H_ */
