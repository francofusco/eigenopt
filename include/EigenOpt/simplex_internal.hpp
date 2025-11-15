#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace EigenOpt {

namespace simplex {

namespace internal {

/// Auxiliay structure to store the domain of a variable.
/** Stores information about the sign that a decision variable can have:
  * - Variables with `non_negative` set to true are supposed to be parameterized
  *     as \f$x=u\f$, \f$u\geq 0\f$.
  * - Variables with `non_positive` set to true are supposed to be parameterized
  *     as \f$x=-v\f$, \f$v\geq0\f$.
  * - Variables with both `non_positive` and `non_negative` set to false are
  *     supposed to be parameterized as \f$x=u-v\f$, \f$u\geq0\f$ and
  *     \f$v\geq0\f$.
  * - Variables with both `non_positive` and `non_negative` set to true are
  *     "degenerate".
  *
  * This structure is not meant to be used directly by the user.
  */
struct VariableDomain {
  bool non_negative = false; ///< If true, \f$x\geq0\f$.
  bool non_positive = false; ///< If true, \f$x\leq0\f$.
  int idx = -1; ///< Which constraint (if any) implies the given domain.
};


/// Given a set of inequality constraints, deduce the domain of the decision variables.
/** The simplex method operates on positive variables. To overcome this
  * limitation, one can perform the substitution \f$x=u-v\f$, where both \f$u\f$
  * and \f$v\f$ are positive. However, some constraints may directly limit the
  * domain of a variable. As an example, consider the constraints
  * \f$-4x_1 \leq -8\f$ and \f$3*x_2 \leq -12\f$. They can be simplified to
  * \f$x_1\geq 2\f$ and \f$x_2\leq -4\f$. It must be noted that \f$x_1\f$ cannot
  * be negative, and \f$x_2\f$ cannot be positive. It is therefore not necessary
  * to introduce a couple of variables for each of these. Instead, one could
  * parameterize them just as \f$x_1=u_1\f$ and \f$x_2=-v_2\f$, with
  * \f$u_1\geq 0\f$ and \f$v_2\geq 0\f$. This function scans the constraints and
  * looks for these situations, storing information about the signs a variable
  * can have. It is also able to detect impossible constraints such as pairs
  * like \f$x \geq 10\f$ and \f$x \leq -5\f$, halting immediately the
  * optimization.
  *
  * @param[in] C Inequality constraints matrix.
  * @param[in] d Inequality constraints vector.
  * @param[in] small_number Tolerance to detect near-zero values and trat them
  *   as 0.
  * @param[out] domains List of domains - one per variable.
  * @param[out] halt_reason If the function returned false, this message
  *   explains why.
  * @return false if impossible constraints were detected, true otherwise.
  */
template<class Scalar>
bool deduce_variables_domains(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& d,
  const Scalar& small_number,
  std::vector<VariableDomain>& domains,
  std::string& halt_reason
);


/// Calculate a transformation matrix so that \f$\bm{x}=\bm{T}\bm{w}\f$, \f$w \geq 0\f$.
/** Decision variables can be parameterized as either: \f$x=u-v\f$, \f$x=u\f$ or
  * \f$x=-v\f$, with \f$u\geq 0\f$ and \f$v\geq 0\f$. When parameterizing
  * multiple variables, it is convenient to express all transformations at once
  * using a matrix \f$\bm{T}\f$. As an example, consider \f$x_1=-v_1\f$,
  * \f$x_2=u_2-v2\f$ and \f$x_3=u_3-v_3\f$. The parameterization can be written
  * in matrix form as \f$\bm{x} = \bm{T}\bm{w}\f$:
  * \f{equation}{
  * \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
  * \begin{bmatrix}
  *   -1 &   &    &   &    \\
  *      & 1 & -1 &   &    \\
  *      &   &    & 1 & -1 \\
  * \end{bmatrix}
  * \begin{bmatrix} v_1 \\ u_2 \\ v_2 \\ u_3 \\ v_3 \end{bmatrix}
  * \f}
  *
  * This function computes the transformation matrix \f$\bm{T}\f$.
  *
  * @tparam Scalar Type used for calculations in Eigen matrices and vectors.
  * @param domains List of variable domains, telling how each variable should
  *   be expressed among the three alternatives.
  * @return The transformation matrix \f$\bm{T}\f$.
  */
template<class Scalar>
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> transformation_matrix_from_domains(
  const std::vector<VariableDomain>& domains
);


/// Calculate a transformation matrix so that \f$\bm{x}=\bm{T}\bm{w}\f$, \f$w \geq 0\f$.
/** This function is a convenience that chains a call to
  * deduce_variables_domains() and transformation_matrix_from_domains().
  *
  * @see deduce_variables_domains()
  * @see transformation_matrix_from_domains()
  *
  * @param[in] C Inequality constraints matrix.
  * @param[in] d Inequality constraints vector.
  * @param[in] small_number Tolerance to detect near-zero values and trat them
  *   as 0.
  * @param[out] T Transformation matrix.
  * @param[out] halt_reason If the function returned false, this message
  *   explains why.
  * @return false if impossible constraints were detected, true otherwise.
  */
template<class Scalar>
bool transformation_matrix(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& d,
  const Scalar& small_number,
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& T,
  std::string& halt_reason
);


/// Create a Simplex Tableau given a set of inequality constraints.
/** This function creates a Simplex Tableau, given a set of inequality
  * constraints in the form \f$\bm{C}\bm{x}\leq\bm{d}\f$ and implicitly assuming
  * that \f$\bm{x}\geq\bm{0}\f$.
  *
  * The Simplex method works as follows:
  * - For each inequalty such that \f$d_i \geq 0\f$, create a new equality
  *   constraint \f$\bm{c}_i^T \bm{x} + s_i = d_i\f$, where \f$s_i\geq 0\f$ is a
  *   slack variable.
  * - For each inequalty such that \f$d_i < 0\f$, create a new equality
  *   constraint \f$-\bm{c}_i^T \bm{x} - s_i + a_i = -d_i\f$, where
  *   \f$s_i \geq 0\f$ is a slack variable and \f$a_i \geq 0\f$ is an artificial
  *   variable.
  *
  * The coefficients are gathered in a matrix (the Tableau) with size
  * \f$(n_v+m+n_a+1, m+1)\f$ - where \f$n_v\f$ is the number of variables in the
  * original problem, \f$m\f$ is the number of inequality constraints (and of
  * slack variables) and \f$n_a\f$ is the nuber of artificial variables.
  *
  * The tableau looks as follows:
  * \f{equation}{
  * \begin{array}{ccc|c}
  *   \bm{\Sigma}\bm{C} & \bm{\Sigma} & \bm{S} & \bm{\Sigma}\bm{d} \\
  *   \hline
  *   \bm{0}^T & \bm{0}^T & \bm{0}^T & 0
  * \end{array}
  * \f}
  * Where \f$\bm{\Sigma}\f$ is a diagonal matrix such that the i-th diagonal
  * entry equals \f$sign(d_i)\f$. \f$\bm{C}\f$ and \f$\bm{d}\f$ are the original
  * constraints matrix and vector. \f$\bm{S}\f$ is a "selection matrix", such that
  * \f$\bm{S}(i,j) = 1\f$ if the constraint \f$i\f$ includes the artificial
  * variable \f$a_j\f$.
  *
  * Furthermore, a basis of \f$m\f$ variables (called "basic-variables") is
  * chosen, so that the equality constraints can be expressed as
  * \f$\bm{M}*\bm{x}_n + \bm{x}_b = \bm{\delta}\f$, with \f$\bm{x}_b\f$ the set
  * of basic variables and \f$\bm{x}_n\f$ the set of non-basic variables. In
  * the creation step of the tableau, the basic variables are always the set of
  * artificial variables plus all slack variables \f$s_i\f$ for which
  * \f$d_i \geq 0\f$.
  *
  * @param[in] C Inequality constraints matrix.
  * @param[in] d Inequality constraints vector.
  * @param[out] tableau Matrix with size \f$(n_v+m+n_a+1, m+1)\f$ - it is
  *   resized as needed, and all its content erased beforehand.
  * @param[out] basic_variables A vector of size \f$m\f$, such that
  *   `basic_variables[row]` tells the index of the active variable for the
  *   given row (there is one basic variable per row). It is resized as needed
  *   and overwritten.
  */
template<class Scalar>
void create_tableau(
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d,
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables
);


/// Use Gaussian elimination on the last row of the tableau.
/** Given a Tableau, run a Gaussian elimination step to make sure that, for
 *  each basic variable, its coefficient in the bottom row becomes 0.
 *
 *  @param[in, out] tableau A Simplex Tableau. It must be in a standard form,
 *    i.e., for each row exactly one basic variable has its coefficient set to
 *    1 (the other basic variables having coefficient 0 in that row). It will
 *    be modified in-place.
 *  @param[in] basic_variables List of basic variables, such that
 *    `basic_variables[row]` tells which is the basic variables associated to
 *    that row of the Tableau.
 */
template<class Scalar>
void eliminate_objective(
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& tableau,
  const std::vector<int>& basic_variables
);


/// Perform a pivot operation between a basic and a non-basic variable.
/** Given a Tableau in standard form, this function runs a simple normalization
 *  step followed by Gaussian elimination.
 *
 *  The normalization step will divide the target row by the coefficient of the
 *  entering variable. It will then use Gaussian elimination to nullify the
 *  coefficient of the entering variable in all other rows. The bottom row is
 *  not modified by this function, to increase versatility. If needed, one can
 *  either call `eliminate_objective()` after a call to `pivot()` to ensure that
 *  all coefficients in the bottom row are processed as expected, or "manually"
 *  eliminate the coefficients as needed.
 *
 *  @warning For this operation to make sense, the Tableau must start in a
 *  valid state, defined by the "rules" of the Simplex method. Furthermore,
 *  the coefficient of the entering variable in the target row must be
 *  positive. Finally, the entering variable must be part of the non-basic set,
 *  and the leaving variable must be part of the basic set. These preconditions
 *  are not checked in this function and if not satisfied will lead to
 *  undefined behavior.
 *
 *  @param[in, out] tableau A Simplex Tableau. It will be modified in-place.
 *  @param[in] entering_variable Index of the non-basic variable that will
 *    enter the basic set.
 *  @param[in] leaving_variable Index of the basic variable that will leave
 *    the basic set.
 */
template<class Scalar>
void pivot(
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  int entering_variable,
  int leaving_variable
);


/// Perform successive pivot operations until a termination condition is met.
/** Given a Tableau in standard form, this function will perform a series of
 *  pivot operations to minimize the associated objective.
 *
 *  At each iteration, the entering and leaving variables are selected
 *  according to the rules of the Simplex method:
 *  - The entering variable is the one whose coefficient in the bottom row is
 *    the most negative one.
 *  - The leaving variable is the one for which the ratio between the
 *    coefficient in the rightmost column and the one in the entering column is
 *    the smallest positive one.
 *  If no entering variable can be found (because all coefficients are
 *  non-negative), an optimal solution has been found. If no leaving variable
 *  can be determined (because all coefficients in the entering column are
 *  non-positive) then the problem is unbounded.
 *
 *  After selection of the pivoting variables, a Gaussian elimination step has
 *  to be performed to set to zero all coefficients in the entering column
 *  except that of the leaving row (which will be normalized to 1).
 *
 *  The two steps (selection of the pivoting variables and Gaussian
 *  elimination) are performed in succession until a termination condition has
 *  been found.
 *
 *  @param[in, out] tableau A Simplex Tableau. It will be modified in-place.
 *    @warning For the algorithm to work as intended, the Tableau must be
 *    valid and already in its reduced form. This precondition is not checked
 *    and if not respected leads to undefined behavior.
 *  @param[in, out] basic_variables List of basic variables. It will be
 *    modified in-place.
 *  @param[in] small_number Tolerance to detect near-zero values and trat them
 *    as 0.
 *  @param[out] halt_reason This message explains (human-readable text) why the
 *    function returned.
 *  @return false if the problem is unbounded, true if the Simplex method
 *    halted successfully.
 */
template<class Scalar>
bool simplex(
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables,
  const Scalar& small_number,
  std::string& halt_reason
);


/// Solve a minimization problem using the two-steps Simplex method.
/** Given a Tableau in standard form (except for the last row, which should be
 *  set to zero), this function will first try to find a feasible point that
 *  satisfies all inequality constraints and then perform successive pivot
 *  operations to reach an optimal solution.
 *
 *  The algorithm starts by adding a unit weight to each artificial variable,
 *  then eliminating all weights to obtain the gradient of the objective
 *  function in terms of non-basic variables. Standard pivoting operations are
 *  then performed to minimize the value of the artificial variables. If the
 *  solution has at least one non-zero artificial variable, then the problem is
 *  infeasible and the function returns.
 *
 *  If after the first step all artificial variables are set to zero, the
 *  algorithm checks if any artificial variable is still in the active set. If
 *  that is the case, it swaps them with non-basic, non-artificial ones.
 *  Artificial variables are then removed from the Tableau entirely. The
 *  objective coefficients are then copied into the bottom row of the Tableau,
 *  and a step of Gaussian elimination is performed to ensure that the Tableau
 *  is in standard form. Pivoting is then performed until the problem is solved
 *  or found to be unbounded.
 *
 *  @param[in] objective Coefficients of the decision variables in the objective
 *    function.
 *  @param[in, out] tableau A Simplex Tableau. It will be modified in-place.
 *    @warning For the algorithm to work as intended, the Tableau must be
 *    valid, with the bottom row set to zero. This precondition is not checked
 *    and if not respected leads to undefined behavior.
 *  @param[in, out] basic_variables List of basic variables. It will be
 *    modified in-place.
 *  @param na Number of artificial variables in the Tableau (the number of
 *    working and slack variables will be calculated automatically).
 *  @param[in] small_number Tolerance to detect near-zero values and trat them
 *    as 0.
 *  @param[out] halt_reason This message explains (human-readable text) why the
 *    function returned.
 *  @return false if the problem is infeasible, true otherwise.
 */
template<class Scalar>
bool two_steps_method(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& objective,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables,
  unsigned int na,
  const Scalar& small_number,
  std::string& halt_reason
);


/// Solve a minimization problem using the penalty Simplex method.
/** Given a Tableau in standard form (except for the last row, which should be
 *  set to zero), this function will try to simultaneously find the optimum
 *  while heavily penalizing constraint infringiment.
 *
 *  The algorithm starts by copying the objective coefficients into the bottom
 *  row of the Tableau, and then adding a large penalty to each artificial
 *  variable. It then eliminates all weights associated to basic variables to
 *  obtain the gradient of the objective function in terms of non-basic
 *  variables. Standard pivoting operations are then performed to minimize the
 *  value of the artificial variables and of the objective simultaneously. If
 *  the solution has at least one non-zero artificial variable, then the
 *  problem is infeasible, otherwise an optimum has been found.
 *
 *  @param[in] objective Coefficients of the decision variables in the objective
 *    function.
 *  @param[in, out] tableau A Simplex Tableau. It will be modified in-place.
 *    @warning For the algorithm to work as intended, the Tableau must be
 *    valid, with the bottom row set to zero. This precondition is not checked
 *    and if not respected leads to undefined behavior.
 *  @param[in, out] basic_variables List of basic variables. It will be
 *    modified in-place.
 *  @param na Number of artificial variables in the Tableau (the number of
 *    working and slack variables will be calculated automatically).
 *  @param[in] small_number Tolerance to detect near-zero values and trat them
 *    as 0.
 *  @param[in] large_number Penalty to be given to artificial variables. It
 *    should be set to a large enough value, so that setting artificial
 *    variables to zero takes the precedence over optimizing the objective.
 *  @param[out] halt_reason This message explains (human-readable text) why the
 *    function returned.
 *  @return false if the problem is infeasible, true otherwise.
 */
template<class Scalar>
bool penalty_method(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& objective,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables,
  unsigned int na,
  const Scalar& small_number,
  const Scalar& large_number,
  std::string& halt_reason
);

} // namespace internal

} // namespace simplex

} // namespace EigenOpt

#include "simplex_internal.hxx"
