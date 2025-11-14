#pragma once

#include <Eigen/Dense>

// Simple macro to define VectorXs and MatrixXs.
#define EigenOptTypedefs(ScalarType)\
typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> VectorXs;\
  typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic> MatrixXs;
