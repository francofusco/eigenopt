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
