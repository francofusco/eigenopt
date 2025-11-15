#pragma once

#include <Eigen/Dense>


namespace EigenOpt {

/// Return all solutions to a linear system, using a kernel-based parameterization.
/** Given a system \f$\bm{A}\bm{x}=\bm{b}\f$, this function tries to solve it by
  * parameterizing \f$ \bm{x} \f$ as
  * \f$ \bm{x} = \bm{x}_{eq} + \bm{Z}\bm{y} \f$
  * where \f$ \bm{x}_{eq} \f$ is the minimum-norm solution to the system in the
  * least squares sense, and \f$ \bm{Z} \f$ is a basis of the kernel of
  * \f$ \bm{A} \f$, i.e., such that \f$ \bm{A}\bm{Z}=\bm{0} \f$. The function
  * uses a Singular Value Decomposition to compute both \f$ \bm{x}_{eq} \f$ and
  * the kernel of \f$ \bm{A} \f$.
  * @param[in] A Matrix of coefficients of the left-hand-side of the linear
  *   system.
  * @param[in] b Vector of coefficients of the right-hand-side of the linear
  *   system.
  * @param[out] Z Projection matrix into the kernel of \f$ \bm{A} \f$, such that
  *   \f$ \bm{A}\bm{Z}=\bm{0} \f$. Note that if the solution to the system is
  *   unique, this matrix will have zero columns.
  * @param[out] xeq Minimum-norm solution to the system
  *   \f$\bm{A}\bm{x}=\bm{b}\f$. Note that this is a solution in the
  *   least-squares sense. To check if the solution is exact, use, e.g.,
  *   `(A*xeq-b).isZero(tolerance)`.
  */
template<class Scalar, class D1, class D2>
void svd_projection(
  const Eigen::MatrixBase<D1>& A,
  const Eigen::MatrixBase<D2>& b,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& Z,
  Eigen::Matrix<Scalar,Eigen::Dynamic,1>& xeq
);


/// Return all solutions to a linear system, using a kernel-based parameterization.
/** Given a system \f$\bm{A}\bm{x}=\bm{b}\f$, this function tries to solve it by
  * parameterizing \f$ \bm{x} \f$ as
  * \f$ \bm{x} = \bm{x}_{eq} + \bm{Z}\bm{y} \f$
  * where \f$ \bm{x}_{eq} \f$ is the minimum-norm solution to the system in the
  * least squares sense, and \f$ \bm{Z} \f$ is a basis of the kernel of
  * \f$ \bm{A} \f$, i.e., such that \f$ \bm{A}\bm{Z}=\bm{0} \f$. The function
  * uses QR factorization to compute both \f$ \bm{x}_{eq} \f$ and the kernel of
  * \f$ \bm{A} \f$.
  * @param[in] A Matrix of coefficients of the left-hand-side of the linear
  *   system.
  * @param[in] b Vector of coefficients of the right-hand-side of the linear
  *   system.
  * @param[out] Z Projection matrix into the kernel of \f$ \bm{A} \f$, such that
  *   \f$ \bm{A}\bm{Z}=\bm{0} \f$. Note that if the solution to the system is
  *   unique, this matrix will have zero columns.
  * @param[out] xeq Minimum-norm solution to the system
  *   \f$\bm{A}\bm{x}=\bm{b}\f$. Note that this is a solution in the
  *   least-squares sense. To check if the solution is exact, use, e.g.,
  *   `(A*xeq-b).isZero(tolerance)`.
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
