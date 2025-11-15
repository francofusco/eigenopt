#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>


namespace EigenOpt {

namespace simplex {


/// Solve a constrained linear optimization problem.
/** This function uses the Simplex method to solve the problem:
 *  \f{equation}{
 *   \min_{\bm{x}} \bm{f}^T \bm{x}
 *   \quad\text{subject to:}\quad
 *   \begin{cases}
 *     \bm{A} \bm{x} = \bm{b} \\
 *     \bm{C} \bm{x} \leq \bm{d}
 *   \end{cases}
 * \f}
 *  where \f$ \bm{f} \f$ is a weight vector, the matrix \f$ \bm{A} \f$ and
 *  vector \f$ \bm{b} \f$ define a set of equality constraints, and the matrix
 *  \f$ \bm{C} \f$ and vector \f$ \bm{d} \f$ define a set of inequality
 *  constraints.
 *
 * To obtain the solution, several steps are needed:
 * 1. Equality constraints are "removed" using
 *      \f$ \bm{x} = \bm{x}_{eq} + \bm{Z}\bm{y} \f$, where \f$ \bm{x}_{eq} \f$
 *      is a particular solution to the equalities and \f$ \bm{Z} \f$ is a basis
 *      of the kernel of \f$ \bm{A} \f$. By substituting into the objective and
 *      inequality constraints, the problem becomes
 *      \f$ \min_{\bm{y}} \bm{f}_y^T \bm{y} \f$ subject to
 *      \f$ \bm{C}_y \bm{y} \leq \bm{d}_y \f$, where
 *      \f$ \bm{f}_y = \bm{f}\bm{Z} \f$, \f$ \bm{C}_y = \bm{C}\bm{Z} \f$ and
 *      \f$ \bm{d}_y = \bm{d} - \bm{C}\bm{x}_{eq} \f$.
 * 2. New variables are then introduced:
 *  - For each coordinates \f$ y_i \f$, two variables \f$ y_i^{(+)} \geq 0 \f$
 *    and \f$ y_i^{(-)} \geq 0 \f$ are introduced. The substitution
 *    \f$ y_i = y_i^{(+)} - y_i^{(-)}\f$ is then performed.
 *  - Each inequality \f$ \bm{c}_i^T \bm{y} \leq d_i \f$ with
 *    \f$ d_i \geq 0 \f$ is transformed in an equality by instroducing a slack
 *    variable \f$ s_i \geq 0 \f$. The new constraint is therefore
 *    \f$ \bm{c}_i^T \bm{y} + s_i = d_i \f$.
 *  - Each inequality \f$ \bm{c}_i^T \bm{y} \leq d_i \f$ with
 *    \f$ d_i < 0 \f$ is transformed in an equality by instroducing a slack
 *    variable \f$ s_i \geq 0 \f$ and an artificial variable \f$ p_i \geq 0 \f$.
 *    The new constraint is therefore
 *    \f$ - \bm{c}_i^T \bm{y} - s_i + p_i = -d_i \f$.
 * 3. The problem has been transformed into the canonical form expected by the
 *      Simplex method, i.e., all decision variables are non-negative and all
 *      constraints are equalities that can be expressed as
 *      \f$ \bm{M}\bm{x}_{n} + \bm{x}_{b} = \bm{\delta} \f$, where
 *      \f$ \bm{x}_{b} \f$ is a subset of the new decision variables, labelled
 *      as "basic variables" (\f$ \bm{x}_{n} \f$ are all remaining, "non-basic"
 *      variables). Initially, the basic vector consists in all artificial
 *      variables plus the slack variables for the constraint with positive
 *      right-hand side. Pivoting operations can now be performed to solve the
 *      problem.
 *
 *  @param[in] f Vector of objective coefficients.
 *  @param[in] A Equality constraints matrix.
 *  @param[in] b Equality constraints vector.
 *  @param[in] C Inequality constraints matrix.
 *  @param[in] d Inequality constraints vector.
 *  @param[out] x Solution vector (modified only if a solution is actually
 *    found).
 *  @param[out] halt_reason A short feedback message explaining why the
 *    optimization routine halted.
 *  @param[in] small_number A small positive tolerance, used to detect
 *    near-zero values. In practice, a number `n` will be considered positive if
 *    `n > small_number` and negative if `n < -small_number`. Finally, if
 *    ``-small_nuber <= n <= small_number`, then `n` is treates as zero.
 *  @param[in] large_number A "large number". When constraints are not
 *    "trivially feasible" (\f$ \bm{x}=\bm{0} \f$ does not satisfy the
 *    inequality \f$ \bm{C}\bm{x}\leq\bm{d} \f$) it is necessary to first
 *    determine a feasible solution. Two methods are commonly employed:
 *    - A "two-steps" method will ignore the original objective and focus on
 *      finding a feasible point. It will focus on optimizing the original
 *      problem only after a feasible solution has been found.
 *    - A "penalty" method will simultaneosly optimize the original objective
 *      and a penalty function which discourages violating the constraints. If
 *      a feasible solution exists, this method will converge to the optimal
 *      solution of the original problem, given a large enough penalty.
 *    .
 *    The two-steps method is more accurate, but takes slightly longer; to use
 *    it, set `large_number` to a negative value. The penalty method is faster
 *    but requires a user-defined constant; set `large_number` to a positive
 *    value to use this method. The value should be several orders of magnitude
 *    larger than the values in the objective and constraints.
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

/// Solve an inequality-constrained linear optimization problem.
/** @overload
  * @see minimize()
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


/// Solve a constrained linear optimization problem.
/** This function uses the Simplex method to solve the problem:
 *  \f{equation}{
 *   \max_{\bm{x}} \bm{f}^T \bm{x}
 *   \quad\text{subject to:}\quad
 *   \begin{cases}
 *     \bm{A} \bm{x} = \bm{b} \\
 *     \bm{C} \bm{x} \leq \bm{d}
 *   \end{cases}
 * \f}
 *
 *  @see minimize()
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


/// Solve an inequality-constrained linear optimization problem.
/** @overload
  * @see minimize()
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


} // simplex

} // EigenOpt


#include "simplex.hxx"
