#pragma once

#include <EigenOpt/kernel_projection.hpp>
#include <EigenOpt/simplex_debug.hpp>


namespace EigenOpt {

namespace simplex {

namespace internal {

template<class Scalar>
bool deduce_variables_domains(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& d,
  const Scalar& small_number,
  std::vector<VariableDomain>& domains,
  std::string& halt_reason
)
{
  // Some syntactic sugar: function that tells if a number is "almost zero".
  auto zero = [&](const Scalar& v) { return -small_number < v && v < small_number; };

  // Deduce the number of constraints and variables.
  const unsigned int m = C.rows();
  const unsigned int n = C.cols();
  domains = std::vector<VariableDomain>(n);

  // For each row of C, check if a single element is non-zero.
  for(int row=0; row<m; row++) {
    int nzcol = -1;
    for(int col=0; col<n; col++) {
      if(!zero(C(row,col))) {
        if(nzcol == -1) {
          // New candidate.
          nzcol = col;
        }
        else {
          // Indicates that the row contains multiple non-zero elements.
          nzcol = -2;
          break;
        }
      }
    }

    // If no non-zero entry was found in a row of C, it means that a constraint
    // is in the form 0*x<=d. This is a degenerate constraint and it will cause
    // an error.
    if(nzcol==-1) {
      halt_reason = "The constraint matrix has row " + std::to_string(row) + " filled with zeros: the problem is degenerate.";
      return false;
    }

    if(nzcol >= 0) {
      // Found a candidate: check if the constraints implies non-negativity/non-positivity.
      if(C(row,nzcol) < 0 && d(row) <= 0) {
        // Add non-negativity constraint.
        EigenOpt_SIMPLEX_DBG("variable " << nzcol << " has a non-negative constraint (row " << row << ")");
        domains[nzcol].non_negative = true;
        domains[nzcol].idx = row;
      }
      if(C(row,nzcol) > 0 && d(row) <= 0) {
        // Make sure there wasn't a non-negativity constraint already as well.
        if(domains[nzcol].non_negative) {
          halt_reason = "Variable " + std::to_string(nzcol) + " has both non-negativity constraint (row " + std::to_string(domains[nzcol].idx) + ") and non-positivity constraint (row " + std::to_string(row) + ").";
          return false;
        }

        // Add non-positivity constraint.
        EigenOpt_SIMPLEX_DBG("variable " << nzcol << " has a non-positive constraint (row " << row << ")");
        domains[nzcol].non_positive = true;
        domains[nzcol].idx = row;
      }
    }
  }

  return true;
}


template<class Scalar>
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> transformation_matrix_from_domains(
  const std::vector<VariableDomain>& domains
  )
{
  // Typedefs 'VectorXs' and 'MatrixXs' - to make my life easier.
  EigenOptTypedefs(Scalar);

  // Number of variables in the original problem.
  const unsigned int n = domains.size();

  // Count how many working variables we will have:
  // - If a variable can be positive, we add one working variable;
  // - If a variable can be negative, we add one working variable;
  // - If a variable can be both positive and negative, we add two working
  //     variables.
  int nv = 0;
  for(int i=0; i<n; i++) {
    if(!domains[i].non_negative)
      nv++;
    if(!domains[i].non_positive)
      nv++;
  }

  // Create the transformation matrix T.
  MatrixXs T = MatrixXs::Zero(n,nv);

  // Current column of T.
  unsigned int col = 0;

  // For each variable, add a 1 and/or a -1 where needed.
  for(int i=0; i<n; i++) {
    if(!domains[i].non_positive) {
      T(i,col) = 1;
      col++;
    }
    if(!domains[i].non_negative) {
      T(i,col) = -1;
      col++;
    }
  }
  eigen_assert(col==nv && "INTERNAL ERROR WHILE INITIALIZING THE TRANSFORMATION MATRIX T: THE FINAL COLUMN COUNT DOES NOT EQUAL THE NUMBER OF WORKING VARIABLES");

  return T;
}


template<class Scalar>
bool transformation_matrix(
  const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& d,
  const Scalar& small_number,
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& T,
  std::string& halt_reason
  )
{
  // Typedefs 'VectorXs' and 'MatrixXs' - to make my life easier.
  EigenOptTypedefs(Scalar);

  // Deduce the domain of each variable from the constraints, and use them to
  // compute the transformation matrix.
  std::vector<VariableDomain> domains;
  if(!deduce_variables_domains(C, d, small_number, domains, halt_reason)) {
    return false;
  }
  T = transformation_matrix_from_domains<Scalar>(domains);
  return true;
}


template<class Scalar>
void pivot(
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  int entering_variable,
  int leaving_variable
)
{
  // Perform gaussian elimination:
  // 1. Normalize the leaving row by the coefficient of the entering variable.
  tableau.row(leaving_variable).noalias() = tableau.row(leaving_variable) / tableau(leaving_variable, entering_variable);
  // 2. For each other row, make sure the coefficient of the of the entering
  //    variable becomes zero. Note that we do not touch the bottom row - just
  //    for sake of versatility (there are cases in which we only want to
  //    perform the pivot on the upper part of the Tableau).
  for(int row=0; row<tableau.rows()-1; row++) {
    if(row!=leaving_variable) {
      tableau.row(row).noalias() -= tableau(row, entering_variable) * tableau.row(leaving_variable);
    }
  }
}


template<class Scalar>
bool simplex(
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables,
  const Scalar& small_number,
  std::string& halt_reason
)
{
  // Number of constraints and of variables.
  const int m = tableau.rows() - 1;
  const int n = tableau.cols() - 1;

  int entering_col; // index of the entering variable

  // Keep iterating until all objective coefficients become non-negative.
  while(tableau.row(m).head(n).minCoeff(&entering_col) < -small_number) {
    EigenOpt_SIMPLEX_DBG("Entering variable index: " << entering_col);

    // Find the leaving variable: it will be the one for which the ratio:
    //   tableau(r, c) / tableau(r, -1)
    // is the smallest non-negative value (in the expression above, c is the
    // entering column and -1 means "last column").
    int leaving_row = -1;
    Scalar minratio, ratio;
    for(int row = 0; row<m; row++) {
      if(tableau(row, entering_col) > small_number) {
        ratio = tableau(row, n) / tableau(row, entering_col);
        if(leaving_row == -1 || ratio < minratio) {
          leaving_row = row;
          minratio = ratio;
        }
      }
    }

    EigenOpt_SIMPLEX_DBG("Leaving row: " << (leaving_row!=-1?std::to_string(leaving_row):"none"));

    if(leaving_row == -1) {
      halt_reason = "No positive coefficient found in the tableau for the entering variable: the problem is unbounded.";
      return false;
    }

    // Keep track of the new basic variable for the leaving row.
    basic_variables[leaving_row] = entering_col;

    // Perform one step of Gaussian elimination.
    pivot(tableau, entering_col, leaving_row);

    EigenOpt_SIMPLEX_DBG("Tableau after Gaussian elimination:" << std::endl << tableau);

    // Nullify the objective weight in the tableau.
    tableau.row(m) -= tableau(m,entering_col) * tableau.row(leaving_row);

    EigenOpt_SIMPLEX_DBG("Tableau after objective nullification:" << std::endl << tableau);
  }

  return true;
}


template<class Scalar>
void create_tableau(
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C,
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d,
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables
)
{
  // Typedefs 'VectorXs' and 'MatrixXs' - to make my life easier.
  EigenOptTypedefs(Scalar);

  // Store number of variables and contraints.
  const unsigned int m = C.rows();
  const unsigned int n = C.cols();

  // Count how many entries in d are negative; for each one of these,
  // we will need an artifical variable.
  const unsigned int na = std::count_if(d.data(), d.data()+m, [](const Scalar& x){return x<0;});
  EigenOpt_SIMPLEX_DBG("Will use " << na << " artifical variables");

  // Prepare the simplex tableau. The variable dcol is both the total number of
  // variables (n working variables, m slack variables and na artificial
  // variables) and the index of the last column of the tableau.
  const unsigned int dcol = n + m + na;
  tableau = MatrixXs::Zero(m+1, dcol+1);
  basic_variables.resize(m);

  // Fill the tableau row-by-row.
  int ia = 0;
  for(int i=0; i<m; i++) {
    auto row = tableau.row(i);
    if(d(i) < 0) {
      // An artificial variable is needed for this constraint.
      EigenOpt_SIMPLEX_DBG("Adding artificial-row to tableau");
      basic_variables[i] = n + m + ia;
      row.head(n) = -C.row(i);
      row(n + i) = -1;
      row(n + m + ia) = 1;
      row(dcol) = -d(i);
      ia++;
    }
    else {
      // Just use the slack variable as usual.
      EigenOpt_SIMPLEX_DBG("Adding slack-row to tableau");
      basic_variables[i] = n + i;
      row.head(n) = C.row(i);
      row(dcol) = d(i);
      row(n + i) = 1;
    }
  }
}


template<class Scalar>
void eliminate_objective(
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& tableau,
  const std::vector<int>& basic_variables
)
{
  // Perform Guassian elimination on the last row, so that at the end all basic
  // coefficients are set to zero.
  auto last_row = tableau.row(tableau.rows()-1);
  for(unsigned int i=0; i<basic_variables.size(); i++) {
    last_row -= last_row(basic_variables[i]) * tableau.row(i);
  }
}


template<class Scalar>
bool two_steps_method(
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& objective,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables,
  unsigned int na,
  const Scalar& small_number,
  std::string& halt_reason
)
{
  // Calculate the number of slack variables and working variables.
  const unsigned int m = tableau.rows() - 1;
  const unsigned int nv = tableau.cols() - m - na - 1;

  // If there is at least one artificial variable, we need to perform both
  // steps. Otherwise, we can jump directly to the second step.
  if(na > 0) {
    EigenOpt_SIMPLEX_DBG("Adding weights for " << na << " artificial variables");

    // Since we have artificial variables, we need to perform the first step.
    // We start by adding a unit weight to each artificial variable in the
    // objective.
    for(unsigned int i=0; i<m; i++) {
      if(basic_variables[i] >= nv + m) {
        EigenOpt_SIMPLEX_DBG("Setting weight for tableau(" << m << "," << basic_variables[i] << ")");
        tableau(m, basic_variables[i]) = 1;
      }
    }

    EigenOpt_SIMPLEX_DBG("Tableau after adding artifical weights:" << std::endl << tableau);

    // Use Gaussian elimination to update the last row of the tableau, so that
    // the weight of basic variables are all set to zero.
    eliminate_objective(tableau, basic_variables);

    EigenOpt_SIMPLEX_DBG("Tableau after objective elimination:" << std::endl << tableau);

    // Now, run the simplex algorithm as usual.
    if(!simplex(tableau, basic_variables, small_number, halt_reason)) {
      return false;
    }

    EigenOpt_SIMPLEX_DBG("Simplex pivoting completed (Step 1).");

    // After the solution, no artificial variable should be greater than zero.
    for(unsigned int i=0; i<m; i++) {
      if(basic_variables[i] >= nv + m && tableau(i, tableau.cols()-1) > small_number) {
        halt_reason = "After pivoting, one artificial variable is still non-basic (p" + std::to_string(basic_variables[i] - nv - m) + " = " + std::to_string(tableau(i, tableau.cols()-1)) + ")";
        return false;
      }
    }

    // Swap basic artificial variables with non-basic ones.
    for(unsigned int i=0; i<m; i++) {
      // Skip non-artificial variables.
      if(basic_variables[i] < nv + m)
        continue;

      EigenOpt_SIMPLEX_DBG("Looking for candidate to swap with p" + std::to_string(basic_variables[i] - nv - m));

      // Find the first non-basic, non-artificial variable in the current row with non-zero coefficient.
      int candidate = -1;
      for(unsigned int j=0; j<nv+m; j++) {
        if(std::find(basic_variables.begin(), basic_variables.end(), j)==basic_variables.end()) {
          if(tableau(i, j) > small_number || tableau(i, j) < -small_number) {
            candidate = j;
            break;
          }
        }
      }

      // If no such variable is found, something is wrong...
      if(candidate < 0) {
        halt_reason = "After the first step, it was not possible to replace the artificial variable p" + std::to_string(basic_variables[i] - nv - m) + " with another non-basic, non-artificial variable.";
        return false;
      }

      EigenOpt_SIMPLEX_DBG("Candidate: " + std::to_string(candidate));

      // Swap the artificial variable and the non-basic one.
      basic_variables[i] = candidate;
      pivot(tableau, candidate, i);
      EigenOpt_SIMPLEX_DBG("Swapped " << basic_variables[i] << " (artificial, previously basic) and " << candidate << " (non-artificial, previously non-basic)");
      EigenOpt_SIMPLEX_DBG("New Tableau:" << std::endl << tableau);
      eigen_assert(tableau(i, tableau.cols()-1) > -small_number && "CRITICAL ISSUE DETECTED: AFTER SWAPPING ZERO-VALUED BASIC ARTIFICIAL VARIABLE, THE NEW BASIC VARIABLE IS NEGATIVE");
    }

    // We can remove the artificial variables from the tableau.
    tableau.col(nv + m) = tableau.col(tableau.cols()-1);
    tableau.conservativeResize(m+1, nv+m+1);

    // Set objective weights in the bottom row.
    tableau.bottomLeftCorner(1, nv) = objective.transpose();
    tableau.bottomRightCorner(1, m+1).setZero();

    EigenOpt_SIMPLEX_DBG("Tableau after removing artificial variables:" << std::endl << tableau);

    // Use Gaussian elimination to update the last row of the tableau, so that
    // the weight of basic variables are all set to zero (since at the end of
    // the first step, at least one working variable will be basic).
    eliminate_objective(tableau, basic_variables);
  }
  else {
    // Since we do not have any artificial variable, the initial tableau is in
    // a feasible state. Just add the objective coefficients at the bottom of
    // the tableau - no elimination is needed since working variables will all
    // be in the non-basic set!
    tableau.bottomLeftCorner(1, nv) = objective.transpose();
  }

  EigenOpt_SIMPLEX_DBG("Tableau after objective elimination:" << std::endl << tableau);

  // Finish by running the simplex algorithm as usual.
  if(!simplex(tableau, basic_variables, small_number, halt_reason)) {
    return false;
  }

  EigenOpt_SIMPLEX_DBG("Simplex pivoting completed (Step 2).");
  return true;
}






template<class Scalar>
bool penalty_method(
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& objective,
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& tableau,
  std::vector<int>& basic_variables,
  unsigned int na,
  const Scalar& small_number,
  const Scalar& large_number,
  std::string& halt_reason
)
{
  // Calculate the number of slack variables and working variables.
  const unsigned int m = tableau.rows() - 1;
  const unsigned int nv = tableau.cols() - m - na - 1;

  // Copy the objective coefficients for the working variables.
  EigenOpt_SIMPLEX_DBG("Adding objective coefficients for working variables");
  tableau.bottomLeftCorner(1, nv) = objective.transpose();

  // Add penalties for artificial variables.
  EigenOpt_SIMPLEX_DBG("Adding penalties for " << na << " artificial variables");
  for(unsigned int i=0; i<m; i++) {
    if(basic_variables[i] >= nv + m) {
      EigenOpt_SIMPLEX_DBG("Setting weight for tableau(" << m << "," << basic_variables[i] << ")");
      tableau(m, basic_variables[i]) = large_number;
    }
  }

  EigenOpt_SIMPLEX_DBG("Tableau after adding weights:" << std::endl << tableau);

  // Use Gaussian elimination to update the last row of the tableau, so that
  // the weight of basic variables are all set to zero.
  eliminate_objective(tableau, basic_variables);

  EigenOpt_SIMPLEX_DBG("Tableau after objective elimination:" << std::endl << tableau);

  // Now, run the simplex algorithm as usual.
  if(!simplex(tableau, basic_variables, small_number, halt_reason)) {
    return false;
  }

  EigenOpt_SIMPLEX_DBG("Simplex pivoting completed.");

  // After the solution, no artificial variable should be greater than zero.
  for(unsigned int i=0; i<m; i++) {
    if(basic_variables[i] >= nv + m && tableau(i, tableau.cols()-1) > small_number) {
      halt_reason = "After pivoting, one artificial variable is still non-basic (p" + std::to_string(basic_variables[i] - nv - m) + " = " + std::to_string(tableau(i, tableau.cols()-1)) + ")";
      return false;
    }
  }

  return true;
}

} // internal

} // simplex

} // EigenOpt
