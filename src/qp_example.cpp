/** @file qp_example.cpp
  * @brief Example program showing how to use the Quadratic Programming solver.
  * @details This is a very basic example where we use Quadratic Programming
  * to solve the problem:
  * \f{equation}{
  *   \min (x_1 + x_2 - 5)^2 \quad \text{s.t.}\;
  *   \begin{cases}
  *     x_1 - x_2 = 10 \\
  *     x_1 + 4 x_2 \leq 0
  *   \end{cases}
  * \f}
  *
  * The objective and constraints, in matrix form, write as:
  * \f{align}{
  *   \mathrm{\mathbf{Q}} &= \begin{bmatrix} 1 & 1 \end{bmatrix}
  *   &&
  *   \mathrm{\mathbf{r}} = \begin{bmatrix} 5 \end{bmatrix}
  *   \\
  *   \mathrm{\mathbf{A}} &= \begin{bmatrix} 1 & -1 \end{bmatrix}
  *   &&
  *   \mathrm{\mathbf{b}} = \begin{bmatrix} 10 \end{bmatrix}
  *   \\
  *   \mathrm{\mathbf{C}} &= \begin{bmatrix} 1 & 4 \end{bmatrix}
  *   &&
  *   \mathrm{\mathbf{d}} = \begin{bmatrix} 0 \end{bmatrix}
  * \f}
  */
#include <EigenOpt/quadratic_programming.hpp>
#include <Eigen/Dense>
#include <iostream>

int main(int argc, char** argv) {
  /* Solve: min (x1 + x2 - 5)^2
   * Such that: x1 - x2 = 10
   *       and: x1 + 4*x2 <= 0
   */
  // Objective and constraints in matrix form.
  Eigen::MatrixXd Q(1, 2);  Q << 1, 1;
  Eigen::VectorXd r(1);     r << 5;
  Eigen::MatrixXd A(1, 2);  A << 1, -1;
  Eigen::VectorXd b(1);     b << 10;
  Eigen::MatrixXd C(1, 2);  C << 1, 4;
  Eigen::VectorXd d(1);     d << 0;
  double tolerance = 1e-6;

  // Create the solver and setup the problem.
  namespace qp = EigenOpt::quadratic_programming;
  qp::Solver<double> solver(Q, r, tolerance);
  solver.setConstraints(A, b, C, d);

  // Solve the problem.
  Eigen::VectorXd x;
  solver.solve(x);
  std::cout << "Solution: " << x.transpose() << std::endl;
  // Prints: "Solution: 7.5  -2.5"

  return 0;
}
