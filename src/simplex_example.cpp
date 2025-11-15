/** @file simplex_example.cpp
  * @brief Example program showing how to use the Simplex solver.
  * @details This is a very basic example where we use the Simplex method to
  * solve the problem:
  * \f{equation}{
  *   \min -x_1+x_2 \quad \text{s.t.}\;
  *   \begin{cases}
  *     -4x_1 -x_2 \leq -5 \\
  *     x_1 - 4x_2 \leq -3 \\
  *     2x_1 - x_2 \leq 8 \\
  *     x_1 \geq 0 \\
  *     x_2 \geq 0
  *   \end{cases}
  * \f}
  *
  * The objective and constraints, in matrix form, write as:
  * \f{equation}{
  *   \mathrm{\mathbf{f}} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
  *   \quad
  *   \mathrm{\mathbf{C}} = \begin{bmatrix} -4 & -1 \\ 1 & -4 \\ 2 & -1 \\ -1 & 0 \\ 0 & -1 \end{bmatrix}
  *   \quad
  *   \mathrm{\mathbf{d}} = \begin{bmatrix} -5 \\ -3 \\ 8 \\ 0 \\ 0 \end{bmatrix}
  * \f}
  */
#include <EigenOpt/simplex.hpp>
#include <Eigen/Dense>
#include <iostream>

int main(int argc, char** argv) {
  /* Solve: min - x1 + x2
   * Such that: -4 x1  - x2 <= -5
   *               x1 -4 x2 <= -3
   *             2 x1  - x2 <= 8
   *                 x1, x2 >=  0
   */
  // Objective and constraints in matrix form.
  Eigen::VectorXd f(2);     f << -1, 1;
  Eigen::MatrixXd C(5, 2);  C << -4,-1,
                                  1,-4,
                                  2,-1,
                                 -1, 0,
                                  0,-1;
  Eigen::VectorXd d(5);     d << -5, -3, 8, 0, 0;
  double tolerance = 1e-6;

  // Solve the problem.
  Eigen::VectorXd x;
  std::string message;
  EigenOpt::simplex::minimize(f, C, d, x, message, tolerance);
  std::cout << "Solution: " << x.transpose() << std::endl;
  // Prints: "Solution: 5  2"

  return 0;
}
