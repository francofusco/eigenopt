Main Page {#mainpage}
=========

Sources are available on
<a href="https://github.com/francofusco/eigenopt">
  <img src="GitHub_Logo.png" alt="GitHub" style="height:30px;">
</a>


During my PhD and postdoc, I became quite well-acquainted with numerical optimization. For some round-tables with fellow students, I prepared some slides and codes to illustrate how some optimization algorithms work. While most of the code was rather unpolished, the Simplex and QP optimization routines were acceptably written, and I decided to gather them here for future reference - and showcase my ability to code :computer:

For now, I included only two solvers - one for linear programming, one for quadratic programming - but I will hopefully have the time in the future to polish others I had created and add them as well. Below are two examples, one using `EigenOpt::simplex::minimize()` and the other featuring `EigenOpt::quadratic_programming::Solver`. You can find more details on them in the respective documentation.

simplex_example.cpp:

```c++
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
```


qp_example.cpp:

```c++
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
```

A more interesting example can be found in example_pixel_robot.cpp, which contains the code to create a very simple simulator for a planar mobile robot that has to move in a rather cluttered environment. Quadratic Programming is used to find the command to be sent to the robot so that it tracks a target (the mouse) while avoiding obstacles.

![Target-following robot](qp_bot.png)
