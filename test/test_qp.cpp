#include <EigenOpt/quadratic_programming.hpp>
#include <fstream>
#include <gtest/gtest.h>


std::string TESTS_DIR;


class QuadraticProgrammingTestFixture : public ::testing::TestWithParam<int> {
protected:
  void SetUp() override {
    const std::string problem_file = TESTS_DIR + "/qp_" + std::to_string(GetParam()) + ".txt";
    std::ifstream file(problem_file);
    if(!file.good())
      throw std::runtime_error("Failed to read test file " + problem_file);

    // Read if the problem is feasible.
    std::string feasible_str;
    file >> feasible_str;
    feasible = feasible_str == std::string("True");

    // Read number of variables in the problem.
    int nv, no, ne, ni;
    file >> nv >> no >> ne >> ni;

    Q.resize(no, nv);
    r.resize(no);
    A.resize(ne, nv);
    b.resize(ne);
    C.resize(ni, nv);
    d.resize(ni);
    x.resize(nv);

    // Read objective.
    for(unsigned int row=0; row<no; row++) {
      for(unsigned int col=0; col<nv; col++) {
        file >> Q(row, col);
      }
    }

    for(unsigned int i=0; i<no; i++) {
      file >> r(i);
    }

    // Read equality constraints.
    for(unsigned int row=0; row<ne; row++) {
      for(unsigned int col=0; col<nv; col++) {
        file >> A(row, col);
      }
    }

    for(unsigned int i=0; i<ne; i++) {
      file >> b(i);
    }

    // Read inequality constraints.
    for(unsigned int row=0; row<ni; row++) {
      for(unsigned int col=0; col<nv; col++) {
        file >> C(row, col);
      }
    }

    for(unsigned int i=0; i<ni; i++) {
      file >> d(i);
    }

    // Read solution vector.
    for(unsigned int i=0; i<nv; i++) {
      file >> x(i);
    }
  }

  bool feasible;
  Eigen::MatrixXd Q;
  Eigen::VectorXd r;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd x;
  static constexpr double solve_tolerance = 1e-9;
  static constexpr double comp_tolerance = 1e-6;
};


TEST_P(QuadraticProgrammingTestFixture, SolveProblem) {
  bool ok;
  Eigen::VectorXd xtest;
  std::string error_message;

  EigenOpt::quadratic_programming::Solver<double> solver(Q, r, solve_tolerance);

  if(A.rows() > 0) {
    ok = solver.setConstraints(A, b, C, d);
  }
  else {
    ok = solver.setConstraints(C, d);
  }

  ASSERT_EQ(feasible, ok);

  if(feasible) {
    ok = solver.solve(xtest);
    ASSERT_TRUE(ok);
    ASSERT_EQ(x.rows(), xtest.rows());
    if(A.rows() > 0) {
      ASSERT_TRUE((A*xtest-b).isZero(solve_tolerance)) << "Some constraints have been violated: abs(A*x-b)= " << (A*xtest-d).cwiseAbs().transpose();
    }
    if(C.rows() > 0) {
      ASSERT_TRUE((C*xtest-d).maxCoeff() <= solve_tolerance) << "Some constraints have been violated: (C*x-d)= " << (C*xtest-d).transpose();
    }
    double obj = (Q*x-r).norm();
    double obj_test = (Q*xtest-r).norm();
    double ftol = comp_tolerance * std::max(1.0, 0.5*(obj+obj_test));
    ASSERT_GE(obj + ftol, obj_test) << "Objective does not match." << std::endl << "x (expected): " << x.transpose() << std::endl << "x (result): " << xtest.transpose();
  }
}


INSTANTIATE_TEST_SUITE_P(QuadraticProgrammingTest, QuadraticProgrammingTestFixture, testing::Range(1, 171));


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  if(argc < 2)
    throw std::runtime_error("Mssing path to tests directory");
  TESTS_DIR = std::string(argv[1]);
  return RUN_ALL_TESTS();
}
