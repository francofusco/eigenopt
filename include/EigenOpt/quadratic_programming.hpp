#pragma once

#include <EigenOpt/typedefs.hpp>

#include <Eigen/Dense>
#include <vector>

namespace EigenOpt {

namespace quadratic_programming {

/// Quadratic Programming solver using active set and null-space projections.
/** This class implements a quadratic programming solver to minimize a given
  * cost function as in:
  * \f{equation}{
  *   \min_{\bm{x}}
  *     \left\| \bm{Q} \bm{x} - \bm{r} \right\|^2
  *   \quad\text{subject to:}\quad
  *   \begin{cases}
  *     \bm{A} \bm{x} = \bm{b} \\
  *     \bm{C} \bm{x} \leq \bm{d}
  *   \end{cases}
  * \f}
  *
  * The reason for using such a specific form for the objective is that it is
  * very well suited for robotics applications, where mathematical models often
  * are written as \f$ \dot{\bm{s}} = \bm{J} \dot{\bm{q}} \f$, where
  * \f$ \bm{s} \f$ is a vector of features while \f$ \bm{q} \f$ are the
  * generalized coordinates of the system - whose derivatives often serve as
  * inputs for the system. Advanced control laws can thus be formulated using
  * quadratic programming.
  *
  * To determine a solution for the problem, the first step is to remove the
  * equalities via a null-space projection strategy. The system of equalities is
  * solved to find a particular solution \f$\bm{x}_{eq}\f$ such that
  * \f$ \bm{A} \bm{x}_{eq} = \bm{b} \f$. Then, an orthonormal basis of the
  * kernel of \f$ \bm{A} \f$ - denoted as \f$ \bm{Z} \f$ - is computed. By
  * definition, its property is that \f$ \bm{A}\bm{Z} = \bm{0} \f$. The original
  * decision vector is thus substituted in the original problem using the
  * parameterization
  * \f{equation}{
  *   \bm{x} = \bm{x}_{eq} + \bm{Z} \bm{y}
  * \f}
  * where \f$ \bm{y} \f$ is a new, lower-dimensional, decision vector. Note that
  * no matter its value, equality constraints will be satisfied. The problem now
  * becomes:
  * \f{equation}{
  *   \min_{\bm{y}}
  *     \left\| \bm{Q}_y \bm{y} - \bm{r}_y \right\|^2
  *   \quad\text{subject to:}\quad
  *   \bm{C}_y \bm{y} \leq \bm{d}_y
  * \f}
  * where:
  * \f{align}{
  *   \bm{Q}_y &\doteq \bm{Q}\bm{Z} \\
  *   \bm{r}_y &\doteq \bm{r} - \bm{Q}\bm{x}_{eq} \\
  *   \bm{C}_y &\doteq \bm{C}\bm{Z} \\
  *   \bm{d}_y &\doteq \bm{d} - \bm{C}\bm{x}_{eq}
  * \f}
  *
  * The problem can be solved using an active-set strategy, whose main steps
  * are:
  * 1. Determine an initial feasible solution that satisfies all inequalities.
  * 2. Parameterize the decision vector as \f$ \bm{y} = \bm{y}_k + \bm{p} \f$,
  *     where \f$ \bm{y}_k \f$ is the current solution and \f$ \bm{p} \f$ is a
  *     "step".
  * 3. Given a set of "active constraints", compute \f$ \bm{p} \f$ so that it
  *     minimizes the objective while enforcing the active constraints.
  * 4. Rather than jumping to the new iterate immediately, now consider the line
  *     parameterization \f$ \bm{y} = \bm{y}_k + \alpha \bm{p} \f$ where
  *     \f$ 0 \leq \alpha \leq 1 \f$. The value of this coefficient is evaluated
  *     as the largest possible value that does not cause any new constraint to
  *     be violated. If the final value is less than 1, it means that a new
  *     constraint has just activated andi t is thus added to the active set. If
  *     \f$ \alpha = 1 \f$, then the full step can be performed without adding
  *     new constraints to the active set.
  * 5. At the new point, it is possible to compute the Lagrange multipliers
  *     associated to the active constraints. If any multiplier is negative, it
  *     means that the constraint does not need to be enforced, and it can be
  *     removed from the active set.
  * 6. Given the new solution and active set, a new iteration is performed from
  *     step 2. The algorithm stops when \f$ \alpha = 1 \f$ and no constraints
  *     deactivate.
  */
template<class Scalar>
class Solver {
public:
  EigenOptTypedefs(Scalar);

  /// Give dimensions for x, Q and r explicitly.
  Solver(int xdim, int rdim, const Scalar& tolerance);

  /// Deduce dimensions from the input matrices.
  template<class D1, class D2>
  Solver(
    const Eigen::EigenBase<D1>& Q,
    const Eigen::EigenBase<D2>& r,
    const Scalar& tolerance
  );

  /// Updates the objective matrix.
  template<class D1, class D2>
  void updateObjective(
    const Eigen::EigenBase<D1>& Q,
    const Eigen::EigenBase<D2>& r
  );

  /// Clear the current active set, preventing warm starts.
  void resetActiveSet();

  /// Removes constraints and clear the active set.
  void clearConstraints();

  /// Add inequality constraints to the problem.
  /** This resets all previous constraints and resets the active set.
    * \warning If a call to this method fails due to infeasible constraints, the
    * problem is left unconstrained. Calling `solve()` will result in solving
    * \f$ \bm{Q} \bm{x} = \bm{r} \f$
    * in the least-squares sense.
    *
    * @param C Inequalities constraints matrix.
    * @param d Inequalities constraints vector.
    * @return true if constraints are feasible, false otherwise.
    */
  template<class D1, class D2>
  bool setConstraints(
    const Eigen::EigenBase<D1>& C,
    const Eigen::EigenBase<D2>& d
  );

  /// Add equality and inequality constraints to the problem.
  /** This resets all previous constraints and resets the active set.
    * \warning If a call to this method fails due to infeasible constraints, the
    * problem is left unconstrained. Calling `solve()` will result in solving
    * \f$ \bm{Q} \bm{x} = \bm{r} \f$
    * in the least-squares sense.
    *
    * @param A Equalities constraints matrix.
    * @param b Equalities constraints vector.
    * @param C Inequalities constraints matrix.
    * @param d Inequalities constraints vector.
    * @return true if constraints are feasible, false otherwise.
    */
  template<class D1, class D2, class D3, class D4>
  bool setConstraints(
    const Eigen::MatrixBase<D1>& A,
    const Eigen::MatrixBase<D2>& b,
    const Eigen::EigenBase<D3>& C,
    const Eigen::EigenBase<D4>& d
  );

  /// Update inequality constraints to the problem.
  /** Existing equality constraints will not be removed. If the constraint
    * dimensions have not changed, the active set will not be reset and
    * feasibility is not tested either. The reason for this method to exists is
    * to help solving multiple similar (but not identical) problems
    * sequentially, warm-starting each one with the information obtained in the
    * previous problem.
    * @param C Inequalities constraints matrix.
    * @param d Inequalities constraints vector.
    * @return true if the constraints did not change dimension, or if they did
    *   and the new ones are feasible.
    */
  template<class D1, class D2>
  bool updateInequalities(
    const Eigen::EigenBase<D1>& C,
    const Eigen::EigenBase<D2>& d
  );

  /// Solve the optimization problem.
  /** @param[out] x Solution of the problem, if one was found.
    * @return true if the optimizaion was successful, false if the problem is
    *   not feasible.
    */
  template<class D>
  bool solve(Eigen::MatrixBase<D>& x);

private:
  /// Solve the problem in the y variable.
  /** @param[out] y Solution in the y variable, if one was found.
    * @return true if the optimizaion was successful, false if the problem is
    *   not feasible.
    */
  template<class D>
  bool solveY(Eigen::MatrixBase<D>& y);

  /// Find an initial solution to start the active-set algorithm.
  /** The following options are considered, in order:
    * - A solution that satisfies all constraints in the active set;
    * - Whatever the last solution was (from a previous call to solveY());
    * - The input value passed by the user, if it has adequate dimension;
    * - A feasible point obtained using the Simplex method.
    * @param y A user-supplied guess.
    * @return true if a feasible point was found, false if the problem has no
    *   solution.
    */
  template<class D>
  bool guess(Eigen::MatrixBase<D>& y);

  /// Initialize the active-set from the current solution.
  /** Once an initial solution has been determined, it is necessary to ensure
    * that the active set contains the constraints "touched" by such solution.
    * @return false if for some reason the current solution violates any
    *   constraint. A constraint is considered violated if
    *   \f$\bm{c}_i\cdot\bm{y} - d_i \f$ exceeds the tolerance.
    */
  bool initActiveSet();

  const Scalar tol; ///< Small tolerance used in calculations.
  const int nx; ///< Number of decision variables.
  const int nr; ///< Number of rows in the objective.
  int ny; ///< Number of variables after removing the equality constraints.
  int mi; ///< Number of inequality constraints.
  int me; ///< number of equality constraints.

  MatrixXs Q; ///< Matrix of coefficients for the objective function.
  VectorXs r; ///< Vector of coefficients for the objective function.

  MatrixXs Z; ///< Matrix that projects into the kernel of the equality constraints matrix.
  VectorXs xeq; ///< A particular solution to the equality constraints.

  MatrixXs Qy; ///< Modified matrix of coefficients for the objective function.
  VectorXs ry; ///< Modified vector of coefficients for the objective function.
  MatrixXs Cy; ///< Modified inequality constraints matrix.
  VectorXs dy; ///< Modified inequality constraints vector.
  VectorXs yu; ///< Unconstrained minimum of the objective.
  VectorXs yk; ///< Current guess of the decision variables.

  MatrixXs Ca; ///< Subset of Cy, corresponding to active constraints.
  VectorXs da; ///< Subset of dy, corresponding to active constraints.
  std::vector<int> active; ///< List of constraints in the active set.
  std::vector<int> inactive; ///< List of constraints not in the active set.
}; // class Solver

} // namespace quadratic_programming

} // namespace EigenOpt


#include "quadratic_programming.hxx"
