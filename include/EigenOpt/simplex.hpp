#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>


namespace EigenOpt {

namespace simplex {

/// Solve an inequality-constrained linear optimization problem.
/** This function uses the Simplex method to solve the problem:
 *    min_x f*x   s.t.  C*x <= d
 *  where f is a weight vector and the matrix C and vector d define a set of
 *  inequality constraints.
 *
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @tparam D1 Matrix type or expression (needed as Eigen uses the CRTP for
 *    templates!) - it can be safely ignored by the user, since it is deduced.
 *  @tparam D2 Like D1.
 *  @tparam D3 Like D1.
 *  @tparam D4 Like D1.
 *  @param[in] f Vector of objective coefficients.
 *  @param[in] C Inequality constraints matrix.
 *  @param[in] d Inequality constraints vector.
 *  @param[out] x Solution vector (modified only if a solution is actually
 *    found).
 *  @param[out] halt_reason A short feedback message explaining why the
 *    optimization routine halted.
 *  @param[in] small_number A small positive tolerance, used to detect
 *    near-zero values. In practice, a number n will be considered positive if
 *    n > small_number and negative if n < -small_number. Finally, if
 *    -small_nuber <= n <= small_number, then n is treates as zero.
 *  @param[in] large_number A "large number". When constraints are not
 *    "trivially feasible" (x=0 does not satisfy the inequality C*x<=d) it is
 *    necessary to first determine a feasible solution. Two methods are
 *    commonly employed:
 *    - A "two-steps" method will ignore the original objective and focus on
 *      finding a feasible point. It will focus on optimizing the original
 *      problem only after a feasible solution has been found.
 *    - A "penalty" method will simultaneosly optimize the original objective
 *      and a penalty function which discourages violating the constraints. If
 *      a feasible solution exists, this method will converge to the optimal
 *      solution of the original problem, given a large enough penalty.
 *    The two-steps method is more accurate, but takes slightly longer; to use
 *    it, set large_number to a negative value. The penalty method is faster
 *    but requires a user-defined constant; set large number to a positive
 *    value to use this method. The value should be several orders of magnitude
 *    larger than the values in the objective and constraints.
 *  @return true if an optimal solution was found, and false otherwise. The
 *    reasons for failure are an unbounded problem or an infeasible constraint
 *    set. The returned message (inside halt_reason) should provide more
 *    details about the reason for a failed optimization.
 */
template<class Scalar, class D1, class D2, class D3, class D4>
bool minimize(
  const Eigen::DenseBase<D1>& f,
  const Eigen::DenseBase<D2>& C,
  const Eigen::DenseBase<D3>& d,
  Eigen::DenseBase<D4>& x,
  std::string& halt_reason,
  const Scalar& small_number,
  const Scalar& large_number=-1
);


/// Solve an inequality-constrained linear optimization problem.
/** This function uses the Simplex method to solve the problem:
 *    max_x f*x   s.t.  C*x <= d
 *  where f is a weight vector and the matrix C and vector d define a set of
 *  inequality constraints.
 *
 *  @see minimize(f, C, d, x, halt_reason, small_number, large_number)
 */
template<class Scalar, class D1, class D2, class D3, class D4>
inline bool maximize(
  const Eigen::DenseBase<D1>& f,
  const Eigen::DenseBase<D2>& C,
  const Eigen::DenseBase<D3>& d,
  Eigen::DenseBase<D4>& x,
  std::string& halt_reason,
  const Scalar& small_number,
  const Scalar& large_number=-1
)
{
  return minimize(-f, C, d, x, halt_reason, small_number, large_number);
}


/// Solve a constrained linear optimization problem.
/** This function uses a projection method in conjunction with the Simplex
 *  method to solve the problem:
 *    min_x f*x   s.t.  A*x = b, C*x <= d
 *  where f is a weight vector, the matrix A and vector b define a set of
 *  equality constraints, and the matrix C and vector d define a set of
 *  inequality constraints.
 *
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @tparam D1 Matrix type or expression (needed as Eigen uses the CRTP for
 *    templates!) - it can be safely ignored by the user, since it is deduced.
 *  @tparam D2 Like D1.
 *  @tparam D3 Like D1.
 *  @tparam D4 Like D1.
 *  @tparam D5 Like D1.
 *  @tparam D6 Like D1.
 *  @param[in] f Vector of objective coefficients.
 *  @param[in] A Equality constraints matrix.
 *  @param[in] b Equality constraints vector.
 *  @param[in] C Inequality constraints matrix.
 *  @param[in] d Inequality constraints vector.
 *  @param[out] x Solution vector (modified only if a solution is actually
 *    found).
 *  @param[out] halt_reason A short feedback message explaining why the
 *    optimization routine halted.
 *  @param[in] small_number @see minimize().
 *  @param[in] large_number @see minimize().
 *  @return true if an optimal solution was found, and false otherwise. The
 *    reasons for failure are an unbounded problem or an infeasible constraint
 *    set. The returned message (inside halt_reason) should provide more
 *    details about the reason for a failed optimization.
 */
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
  const Scalar& large_number=-1
);


/// Solve a constrained linear optimization problem.
/** This function uses the Simplex method to solve the problem:
 *    max_x f*x   s.t.  A*x = b, C*x <= d
 *  where f is a weight vector, the matrix A and vector b define a set of
 *  equality constraints, and the matrix C and vector d define a set of
 *  inequality constraints.
 *
 *  @see minimize(f, A, b, C, d, x, halt_reason, small_number, large_number)
 */
template<class Scalar, class D1, class D2, class D3, class D4, class D5, class D6>
bool maximize(
  const Eigen::MatrixBase<D1>& f,
  const Eigen::MatrixBase<D2>& A,
  const Eigen::MatrixBase<D3>& b,
  const Eigen::MatrixBase<D4>& C,
  const Eigen::MatrixBase<D5>& d,
  Eigen::DenseBase<D6>& x,
  std::string& halt_reason,
  const Scalar& small_number,
  const Scalar& large_number=-1
)
{
  return minimize(-f, A, b, C, d, x, halt_reason, small_number, large_number);
}


} // simplex

} // EigenOpt


#include "simplex.hxx"
