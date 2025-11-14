#pragma once

#include <Eigen/Dense>


namespace EigenOpt {

/// Return all solutions to a linear system, using a kernel-based parameterization.
/** Given a system A*x=b, this function tries to solve it by parameterizing x as
  * x = xeq + Z*y where xeq is the minimum-norm solution to the system in the
  * least squares sense, and Z is a basis of the kernel of A, i.e., such that
  * A*Z = 0. The function uses a Singular Value Decomposition to compute both
  * xeq and the kernel of A.
  * @param[in] A Matrix of coefficients of the left-hand-side of the linear
  *   system.
  * @param[in] b Vector of coefficients of the right-hand-side of the linear
  *   system.
  * @param[out] Z Projection matrix into the kernel of A, such that A*Z=0. Note
  *   that if the solution to the system is unique, this matrix will have zero
  *   columns.
  * @param[out] xeq Minimum-norm solution to the system A*x=b. Note that this is
  *   a solution in the least-squares sense. To check if the solution is exact,
  *   use, e.g., (A*xeq-b).isZero(tolerance).
  */
template<class Scalar, class D1, class D2>
void svd_projection(
  const Eigen::MatrixBase<D1>& A,
  const Eigen::MatrixBase<D2>& b,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& Z,
  Eigen::Matrix<Scalar,Eigen::Dynamic,1>& xeq
);


/// Return all solutions to a linear system, using a kernel-based parameterization.
/** Given a system A*x=b, this function tries to solve it by parameterizing x as
  * x = xeq + Z*y where xeq is a solution to the system in the least squares
  * sense, and Z is a basis of the kernel of A, i.e., such that A*Z = 0. The
  * function uses two QR Decompositions to compute both xeq and the kernel of A.
  * @param[in] A Matrix of coefficients of the left-hand-side of the linear
  *   system.
  * @param[in] b Vector of coefficients of the right-hand-side of the linear
  *   system.
  * @param[out] Z Projection matrix into the kernel of A, such that A*Z=0. Note
  *   that if the solution to the system is unique, this matrix will have zero
  *   columns.
  * @param[out] xeq A solution to the system A*x=b, in the least-squares sense.
  *   To check if the solution is exact, use, e.g., (A*xeq-b).isZero(tolerance).
  */
template<class Scalar, class D1, class D2>
void qr_projection(
  const Eigen::MatrixBase<D1>& A,
  const Eigen::MatrixBase<D2>& b,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& Z,
  Eigen::Matrix<Scalar,Eigen::Dynamic,1>& xeq
);

} // namespace EigenOpt

#include "kernel_projection.hxx"
