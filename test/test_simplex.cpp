#include <EigenOpt/simplex.hpp>
#include <fstream>
#include <gtest/gtest.h>


std::string TESTS_DIR;


class SimplexTestFixture : public ::testing::TestWithParam<int> {
protected:
  void SetUp() override {
    const std::string problem_file = TESTS_DIR + "/lp_" + std::to_string(GetParam()) + ".txt";
    std::ifstream file(problem_file);
    if(!file.good())
      throw std::runtime_error("Failed to read test file " + problem_file);

    // Read if the problem is feasible.
    std::string feasible_str;
    file >> feasible_str;
    feasible = feasible_str == std::string("True");

    // Read number of variables in the problem.
    int n, ne, ni;
    file >> n >> ne >> ni;

    f.resize(n);
    A.resize(ne, n);
    b.resize(ne);
    C.resize(ni, n);
    d.resize(ni);
    x.resize(n);

    // Read objective vector.
    for(unsigned int i=0; i<n; i++) {
      file >> f(i);
    }

    // Read equality constraints.
    for(unsigned int row=0; row<ne; row++) {
      for(unsigned int col=0; col<n; col++) {
        file >> A(row, col);
      }
    }

    for(unsigned int i=0; i<ne; i++) {
      file >> b(i);
    }

    // Read inequality constraints.
    for(unsigned int row=0; row<ni; row++) {
      for(unsigned int col=0; col<n; col++) {
        file >> C(row, col);
      }
    }

    for(unsigned int i=0; i<ni; i++) {
      file >> d(i);
    }

    // Read solution vector.
    for(unsigned int i=0; i<n; i++) {
      file >> x(i);
    }
  }

  void runTest(bool with_penalty);

  bool feasible;
  Eigen::VectorXd f;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  Eigen::VectorXd x;
  static constexpr double penalty = 1e6;
  static constexpr double tolerance = 1e-6;
};


void SimplexTestFixture::runTest(bool with_penalty) {
  bool ok;
  Eigen::VectorXd xtest;
  std::string error_message;

  if(A.rows() > 0) {
    ok = EigenOpt::simplex::minimize(f, A, b, C, d, xtest, error_message, tolerance, with_penalty ? penalty : -1.0);
  }
  else {
    ok = EigenOpt::simplex::minimize(f, C, d, xtest, error_message, tolerance, with_penalty ? penalty : -1.0);
  }

  ASSERT_EQ(feasible, ok);

  if(ok) {
    ASSERT_EQ(x.rows(), xtest.rows());
    double ftol = std::max(tolerance * std::abs(f.dot(x)), tolerance);
    ASSERT_NEAR(f.dot(x), f.dot(xtest), ftol) << "Objective does not match." << std::endl << "x (expected): " << x.transpose() << std::endl << "x (result): " << xtest.transpose();
  }
}


TEST_P(SimplexTestFixture, PenaltyMethod) {
  runTest(true);
}


TEST_P(SimplexTestFixture, TwoStepsMethod) {
  runTest(false);
}


INSTANTIATE_TEST_SUITE_P(SimplexTest, SimplexTestFixture, testing::Range(1, 205));


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  if(argc < 2)
    throw std::runtime_error("Mssing path to tests directory");
  TESTS_DIR = std::string(argv[1]);
  return RUN_ALL_TESTS();
}
