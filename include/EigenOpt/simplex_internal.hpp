#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace EigenOpt {

namespace simplex {

namespace internal {

/// Auxiliay structure to store the domain of a variable.
/** Stores information about the sign a decision variable can have:
 *  - Variables with non_negative set to true are supposed to be parameterized
 *      as x=u, u>0.
 *  - Variables with non_positive set to true are supposed to be parameterized
 *      as x=-v, v>0.
 *  - Variables with both non_positive and non_negative set to false are
 *      supposed to be parameterized as x=u-v, u>0 and v>0.
 *  - Variables with both non_positive and non_negative set to true are
 *      "degenerate".
 *
 *  This structure is not meant to be used directly by the user.
 */
struct VariableDomain {
  bool non_negative = false; ///< If true, x>=0.
  bool non_positive = false; ///< If true, x<=0.
  int idx = -1; ///< Which constraint (if any) implies the given domain.
};


/// Given a set of inequality constraints, deduce the domain of the decision variables.
/** The simplex method operates on positive variables. To overcome this
 *  limitation, one can perform the substitution x = u - v, where both u and v
 *  are positive. However, some constraints may directly limit the domain of a
 *  variable. As an example, consider the constraints -4*x_1 < -8 and
 *  3*x_2 < -12. They can be simplified to x_1>2 and x_2<-4. It must be noted
 *  that x_1 cannot be negative, and x_2 cannot be positive. It is therefore
 *  not necessary to introduce a couple of variables for each of these.
 *  Instead, one could parameterize them just as x_1=u_1 and x_2=-v_2, with
 *  u_1>=0 and v_2>=0. This function scans the constraints and looks for these
 *  situations, storing information about the signs a variable can have. It is
 *  also able to detect impossible constraints such as pairs like x > 10 and
 *  x < -5, halting immediately the optimization.
 *
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @param[in] C Inequality constraints matrix.
 *  @param[in] d Inequality constraints vector.
 *  @param[in] small_number Tolerance to detect near-zero values and trat them
 *    as 0.
 *  @param[out] domains List of domains - one per variable.
 *  @param[out] halt_reason If the function returned false, this message
 *    explains why.
 *  @return false if impossible constraints were detected, true otherwise.
 */
template<class Scalar>
bool deduce_variables_domains(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& d,
  const Scalar& small_number,
  std::vector<VariableDomain>& domains,
  std::string& halt_reason
);


/// Calculate a transformation matrix so that x=T*w, w>=0.
/** Decision variables can be parameterized as either: x=u-v, x=u or x=-v, with
 *  u>=0 and v>=0. When parameterizing multiple variables, it is convenient to
 *  express all transformations at once using a matrix T. As an example,
 *  consider x_1=-v_1, x_2=u_2-v2 and x_3=u_3-v_3. The parameterization can be
 *  written in matrix form as x = T*w:
 *
 *  \verbatim
 *                                 [ v_1 ]
 *    [ x_1 ]   [ -1           ]   [ u_2 ]
 *    [ x_2 ] = [    1 -1      ] * [ v_2 ]
 *    [ x_3 ]   [         1 -1 ]   [ u_3 ]
 *                                 [ v_3 ]
 *  \endverbatim
 *
 *  This function computes the transformation matrix T.
 *
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @param domains List of variable domains, telling how each variable should
 *    be expressed among the three alternatives.
 *  @return The transformation matrix T.
 */
template<class Scalar>
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> transformation_matrix_from_domains(
  const std::vector<VariableDomain>& domains
);


/// Calculate a transformation matrix so that x=T*w, w>=0.
/** This function is a convenience that chains a call to
 *  deduce_variables_domains() and transformation_matrix_from_domains().
 *
 *  @see deduce_variables_domains()
 *  @see transformation_matrix_from_domains()
 *
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @param[in] C Inequality constraints matrix.
 *  @param[in] d Inequality constraints vector.
 *  @param[in] small_number Tolerance to detect near-zero values and trat them
 *    as 0.
 *  @param[out] T Transformation matrix.
 *  @param[out] halt_reason If the function returned false, this message
 *    explains why.
 *  @return false if impossible constraints were detected, true otherwise.
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
 *  constraints in the form C*x<=d and implicitly assuming that x>=0.
 *
 *  The Simplex method works as follows:
 *  - For each inequalty such that d_i >= 0, create a new equality constraint
 *    c_i^T * x + s_i = d_i, where s_i>=0 is a slack variable.
 *  - For each inequalty such that d_i < 0, create a new equality constraint
 *    -c_i^T * x - s_i + a_i = -d_i, where s_i>=0 is a slack variable and
 *    a_i>=0 is an artificial variable.
 *
 *  The coefficients are gathered in a matrix (the Tableau) with size
 *  (nv+m+na+1, m+1) - where nv is the number of variables in the original
 *  problem, m is the number of inequality constraints (and of slack variables)
 *  and na is the nuber of artificial variables.
 *
 *  The tableau looks as follows:
 *
 *  \verbatim
 *   DC  D  A | Dd
 *  ---------+-----
 *    0  0  0 |  0
 *  \endverbatim
 *
 *  Where D is a diagonal matrix such that the i-th diagonal entry equals
 *  sign(d_i). C and d are the original constraints matrix and vector. A is a
 *  "selection matrix", such that A(i,j) = 1 if the constraint i includes the
 *  artificial variable a_j. Note that the tableau is
 *
 *  Furthermore, a basis of m variables (called "basic-variables") is chosen,
 *  so that the equality constraints can be expressed as M*x_n + x_b = r, with
 *  x_b the set of basic variables and x_n the set of non-basic variables. In
 *  the creation step of the tableau, the basic variables are always the set of
 *  artificial variables plus all slack variables s_i for which d_i>=0.
 *
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @param[in] C Inequality constraints matrix.
 *  @param[in] d Inequality constraints vector.
 *  @param[out] tableau Matrix with size (nv+m+na+1, m+1) - it is resized as
 *    needed, and all its content erased beforehand.
 *  @param[out] basic_variables A vector of size m, such that
 *    basic_variables[row] tells the index of the active variable for the
 *    given row (there is one basic variable per row). It is resized as needed
 *    and overwritten.
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
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
 *  @param[in, out] tableau A Simplex Tableau. It must be in a standard form,
 *    i.e., for each row exactly one basic variable has its coefficient set to
 *    1 (the other basic variables having coefficient 0 in that row). It will
 *    be modified in-place.
 *  @param[in] basic_variables List of basic variables, such that
 *    basic_variables[row] tells which is the basic variables associated to
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
 *  either call eliminate_objective() after a call to pivot() to ensure that
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
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
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
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
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
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
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
 *  @tparam Scalar Type used for calculations in Eigen matrices and vectors.
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
