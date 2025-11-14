#pragma once

#include <EigenOpt/typedefs.hpp>
#include <EigenOpt/kernel_projection.hpp>
#include <EigenOpt/simplex_debug.hpp>
#include <EigenOpt/simplex_internal.hpp>


namespace EigenOpt {

namespace simplex {


template<class Scalar, class D1, class D2, class D3, class D4>
bool minimize(
  const Eigen::DenseBase<D1>& _f,
  const Eigen::DenseBase<D2>& _C,
  const Eigen::DenseBase<D3>& _d,
  Eigen::DenseBase<D4>& x,
  std::string& halt_reason,
  const Scalar& small_number,
  const Scalar& large_number
  )
{
  // Typedefs 'VectorXs' and 'MatrixXs' - to make my life easier.
  EigenOptTypedefs(Scalar);

  // Check consistency of the numeric parameters
  eigen_assert(small_number>0 && "PARAMETER 'small_number' MUST BE POSITIVE");

  EigenOpt_SIMPLEX_DBG("Attempting to solve minimization problem with following parameters:");
  EigenOpt_SIMPLEX_DBG("Objective coefficients: " << _f.transpose());
  EigenOpt_SIMPLEX_DBG("C:" << std::endl << _C);
  EigenOpt_SIMPLEX_DBG("d: " << _d.transpose());

  // Cast objective coefficients into dense format & store the number of decision variables.
  VectorXs f = _f;
  int n = f.rows();

  // We allow an empty objective, which means that the objective should be filled with zeros.
  // In this case, the number of decision variables is deduced from the constraint matrix.
  if(n == 0) {
    n = _C.cols();
    eigen_assert(n>0 && "THE PROBLEM DOES NOT HAVE ANY VARIABLE");
    f = VectorXs::Zero(n);
    EigenOpt_SIMPLEX_DBG("Objective coefficients omitted, assuming they are all zero");
  }

  // Make sure dimensions are consistent.
  eigen_assert(_C.rows()==_d.rows() && "C MATRIX AND D VECTOR HAVE DIFFERENT NUMBER OF ROWS");
  eigen_assert(_C.cols()==n && "C MATRIX HAS WRONG NUMBER OF COLUMNS");

  // Since this function does not make prior assumptions of the bounds of the
  // decision variables, a problem with no constraints is ill-defined - the
  // "solution" is to let decision variables be infinite.
  if(_C.rows() == 0) {
    halt_reason = "No constraints given, the problem is ill-defined";
    return false;
  }

  // Store the original constraints, removing degenerate ones (those in the
  // form 0*x<=k, with k >=0) and detecting infeasible ones (0*x<=k<0).
  MatrixXs C(_C.rows(), _C.cols());
  VectorXs d(_d.rows());

  // Remove degenerate constraints in the form 0*x<=k.
  int m = 0;
  for(unsigned int i=0; i<_C.rows(); i++) {
    if(!_C.row(i).isZero(small_number)) {
      // Row i is not degenerate: keep the constraint.
      C.row(m).noalias() = _C.row(i);
      d(m) = _d(i);
      m++;
    }
    else if(_d(i) < 0) {
      // Row i is degenerate and d_i is negative: the problem is infeasible.
      halt_reason = "Found infeasible degenerate constraint (row " + std::to_string(i) + ").";
      return false;
    }
  }

  // Resize the constraints to discard unused rows.
  C.conservativeResize(m, C.cols());
  d.conservativeResize(m);

  EigenOpt_SIMPLEX_DBG("Of the original " << _C.rows() << " constraints, " << m << " were kept:");
  EigenOpt_SIMPLEX_DBG("C:" << std::endl << C);
  EigenOpt_SIMPLEX_DBG("d: " << d.transpose());

  // Obtain the transformation matrix and update the problem.
  MatrixXs T;
  if(!internal::transformation_matrix(C, d, small_number, T, halt_reason)) {
    return false;
  }
  EigenOpt_SIMPLEX_DBG("Transformation matrix T =" << std::endl << T);

  // Modfied constraints and objective.
  VectorXs fs = T.transpose()*f;
  MatrixXs Cs = C*T;
  const unsigned int nv = T.cols();

  // Simplex Tableau.
  MatrixXs tableau;

  // List of basic variables, ordered by row, i.e., the column of the basic
  // variable used in a row is given by basic_variables[row].
  std::vector<int> basic_variables;

  // Fill the upper portion of the tableau.
  internal::create_tableau(Cs, d, tableau, basic_variables);
  EigenOpt_SIMPLEX_DBG("Initial tableau:" << std::endl << tableau);

  // Deduce the number of artificial variables added to the tableau.
  const unsigned int na = tableau.cols() - nv - m - 1;

#ifdef EIGEN_SIMPLEX_DEBUG_ON
  EigenOpt_SIMPLEX_DBG("Basic variables:");
  for(unsigned int i=0; i<m; i++) {
    EigenOpt_SIMPLEX_DBG("- " << basic_variables[i] << " (" << (basic_variables[i]<nv+m ? "slack" : "artificial") << ")");
  }
#endif

  // Solve the problem using either a penalty or a two-steps method.
  if(large_number > 0) {
    if(!internal::penalty_method(fs, tableau, basic_variables, na, small_number, large_number, halt_reason)) {
      return false;
    }
  }
  else {
    if(!internal::two_steps_method(fs, tableau, basic_variables, na, small_number, halt_reason)) {
      return false;
    }
  }

  // Read the solution from the tableau.
  VectorXs xv = VectorXs::Zero(nv);
  for(unsigned int i = 0; i<m; i++) {
    if(basic_variables[i] < nv) {
      xv(basic_variables[i]) = tableau(i, tableau.cols()-1);
    }
  }

  // Project back to the original domain.
  x = T * xv;
  eigen_assert((C*VectorXs(x) - d).maxCoeff() < small_number && "Something went horribly wrong: Simplex optimization was completed 'successfully' but constraints are not respected.");
  halt_reason = "Optimization completed successfully";
  return true;
}


template<class Scalar, class D1, class D2, class D3, class D4, class D5, class D6>
bool minimize(
  const Eigen::MatrixBase<D1>& f,
  const Eigen::MatrixBase<D2>& A,
  const Eigen::MatrixBase<D3>& b,
  const Eigen::MatrixBase<D4>& C,
  const Eigen::MatrixBase<D5>& d,
  Eigen::DenseBase<D6>& x,
  std::string& halt_reason,
  const Scalar& small_number,
  const Scalar& large_number
)
{
  // Typedefs 'VectorXs' and 'MatrixXs' - to make my life easier.
  EigenOptTypedefs(Scalar);

  // Solve A*x = b in least squares sense.
  VectorXs xeq;
  MatrixXs Z;
  #ifdef EigenOpt_SIMPLEX_USE_QR_INSTEAD_OF_SVD
    qr_projection(A, b, Z, xeq);
  #else
    svd_projection(A, b, Z, xeq);
  #endif

  // If A*x=b has no solutions, xeq is the solution in the least squares sense,
  // which we cannot accept in this context.
  if(!(A*xeq-b).isZero(small_number)) {
    halt_reason = "Equality constraints are infeasible.";
    return false;
  }

  EigenOpt_SIMPLEX_DBG("Particular solution for equality constraints: " << xeq.transpose());

  // Check if the equality constraints fully constrain the decision vector.
  if(Z.cols() == 0) {
    // The solution xeq is compatible with all constraints, but we do not have
    // any DOF left: this is the best we can do and there is no point in going
    // further.
    x = xeq;
    halt_reason = "The solution is fully determined by equality constraints";
    return true;
  }

  EigenOpt_SIMPLEX_DBG("Projection matrix into ker(A):" << std::endl << Z);


  // We have more DOFs remaining; use a projection into the kernel of A to
  // obtain the full solution! Z is the projection matrix that maps into the
  // kernel of A, i.e., such that A*Z=0. We can thus parameterize x as:
  // x = xeq + Z*y (with y a "free", lower-dimensional vector) and optimize
  // over y - since for all values of y, equality constraints will be met. The
  // reduced problem becomes:
  //   min_y (Z^T * f)^T * y  s.t.  C*Z*y <= d - C*xeq
  VectorXs y;
  bool ok = minimize(Z.transpose()*f, C*Z, d-C*xeq, y, halt_reason, small_number, large_number);
  if(!ok) {
    halt_reason = "Failed to solve the inequality constrained sub-problem: " + halt_reason;
  }
  else {
    x = xeq + Z*y;
  }
  return ok;
}

} // simplex

} // EigenOpt
