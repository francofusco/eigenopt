#pragma once

#include <EigenOpt/typedefs.hpp>

namespace EigenOpt {

template<class Scalar, class D1, class D2>
void svd_projection(
  const Eigen::MatrixBase<D1>& A,
  const Eigen::MatrixBase<D2>& b,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& Z,
  Eigen::Matrix<Scalar,Eigen::Dynamic,1>& xeq
)
{
  EigenOptTypedefs(Scalar);

  // Perform a SVD on A, making sure to compute the full V to extract ker(A).
  Eigen::JacobiSVD<MatrixXs> svd(A, Eigen::ComputeThinU|Eigen::ComputeFullV);

  // Find the minimum-norm, least square solution to A*x=b.
  xeq = svd.solve(b);

  // SVD is a rank-revealing decomposition, and we can use it to check if we
  // actually have some degrees of freedom left in x.
  if(A.cols() > svd.rank()) {
    // Yes, we have some additional degrees of freedom: extract the kernel of A.
    Z = svd.matrixV().rightCols(svd.cols()-svd.rank());
  }
  else {
    // No, we do not have any degrees of freedom (the system fully determines a
    // value for xeq) and therefore the kernel is "empty".
    Z.resize(A.cols(), 0);
  }
}


template<class Scalar, class D1, class D2>
void qr_projection(
  const Eigen::MatrixBase<D1>& A,
  const Eigen::MatrixBase<D2>& b,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& Z,
  Eigen::Matrix<Scalar,Eigen::Dynamic,1>& xeq
)
{
  EigenOptTypedefs(Scalar);

  // Solve A*x = b in least squares sense (in the sense that it minimizes
  // A*x-b, but it is not necessarily the minimum-norm solution) using a QR
  // decomposition.
  xeq = Eigen::ColPivHouseholderQR<MatrixXs>(A).solve(b);

  // Do a QR decomposition of A^T to extract the orthogonal matrix Q.
  Eigen::ColPivHouseholderQR<MatrixXs> QR(A.transpose());

  // QR is a rank-revealing decomposition, and we can use it to check if we
  // actually have some degrees of freedom left in x.
  if(A.cols() > QR.rank()) {
    // Yes, we have some additional degrees of freedom: extract the kernel of A
    Z = MatrixXs(QR.householderQ()).rightCols(A.cols()-QR.rank());
  }
  else {
    // No, we do not have any degrees of freedom (the system fully determines a
    // value for xeq) and therefore the kernel is "empty".
    Z.resize(A.cols(), 0);
  }
}

} // namespace EigenOpt
