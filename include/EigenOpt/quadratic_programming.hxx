#pragma once

#include <EigenOpt/kernel_projection.hpp>
#include <EigenOpt/simplex.hpp>
#include <EigenOpt/quadratic_programming_debug.hpp>


namespace EigenOpt {

namespace quadratic_programming {


template<class Scalar>
Solver<Scalar>::Solver(int xdim, int rdim, const Scalar& tolerance)
: nx(xdim)
, ny(xdim)
, nr(rdim)
, mi(0)
, me(0)
, tol(tolerance)
{
  EigenOpt_QUADPROG_DBG("Calling constructor with sizes " << xdim << " and " << rdim);
  eigen_assert(nx>0 && "AT LEAST ONE DECISION VARIABLE IS REQUIRED");
  eigen_assert(nr>0 && "AT LEAST ONE OBJECTIVE ROW IS REQUIRED");
  Qy = Q = MatrixXs::Zero(nr, nx);
  ry = r = VectorXs::Zero(nr);
  Cy = MatrixXs::Zero(0, nx);
  dy = VectorXs::Zero(0);
  Z = MatrixXs::Identity(nx, nx);
  yk = yu = xeq = VectorXs::Zero(nx);
  resetActiveSet();
}


template<class Scalar>
template<class D1, class D2>
Solver<Scalar>::Solver(
  const Eigen::EigenBase<D1>& Q,
  const Eigen::EigenBase<D2>& r,
  const Scalar& tolerance
)
: Solver<Scalar>(Q.cols(), Q.rows(), tolerance)
{
  EigenOpt_QUADPROG_DBG("Calling constructor with Q and r");
  updateObjective(Q,r);
}


template<class Scalar>
template<class D1, class D2>
void Solver<Scalar>::updateObjective(
  const Eigen::EigenBase<D1>& Q,
  const Eigen::EigenBase<D2>& r
)
{
  EigenOpt_QUADPROG_DBG("Updating objective");
  eigen_assert(Q.rows()==nr && "Q MATRIX HAS WRONG NUMBER OF ROWS");
  eigen_assert(Q.cols()==nx && "Q MATRIX HAS WRONG NUMBER OF COLUMNS");
  eigen_assert(r.rows()==nr && "R VECTOR HAS WRONG NUMBER OF ROWS");
  this->Q = Q;
  this->r = r;
  EigenOpt_QUADPROG_DBG("Q=" << std::endl << this->Q << std::endl << "and vector r=" << std::endl << this->r);

  // If equality constraints have been set, reduce the problem.
  if(me > 0) {
    if(ny > 0) {
      Qy = this->Q * Z;
      ry = this->r - this->Q * xeq;
    }
    else {
      Qy = MatrixXs::Zero(nr, 0);
      ry = VectorXs::Zero(nr);
    }
  }
  else {
    Qy = Q;
    ry = r;
  }

  EigenOpt_QUADPROG_DBG("Qy=" << std::endl << Qy << std::endl << "ry=" << std::endl << ry);

  if(ny > 0) {
    #ifdef USE_QR_INSTEAD_OF_SVD
      yu = Qy.fullPivHouseholderQr().solve(ry);
    #else
      // yu = Qy.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(ry);
      yu = Qy.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(ry);
    #endif
  }
  else {
    yu = VectorXs::Zero(0);
  }
  EigenOpt_QUADPROG_DBG("Unconstrained minimum:" << std::endl << "y: " << yu.transpose() << std::endl << "x: " << (xeq+Z*yu).transpose());
}


template<class Scalar>
void Solver<Scalar>::resetActiveSet()
{
  EigenOpt_QUADPROG_DBG("Resetting active set");
  Ca.resize(0, ny);
  da.resize(0);
  active.clear();
  active.reserve(mi);
  inactive.resize(mi);
  for(int i=0; i<mi; i++)
    inactive[i] = i;
}


template<class Scalar>
void Solver<Scalar>::clearConstraints()
{
  Z = MatrixXs::Identity(nx, nx);
  xeq = VectorXs::Zero(nx);
  Cy = MatrixXs::Zero(0,ny);
  dy = VectorXs::Zero(0);
  mi = 0;
  me = 0;
  ny = nx;
  resetActiveSet();
  updateObjective(Q, r);
}


template<class Scalar>
template<class D1, class D2>
bool Solver<Scalar>::setConstraints(
  const Eigen::EigenBase<D1>& C,
  const Eigen::EigenBase<D2>& d
)
{
  return setConstraints(MatrixXs(0, nx), VectorXs(0), C, d);
}


template<class Scalar>
template<class D1, class D2, class D3, class D4>
bool Solver<Scalar>::setConstraints(
  const Eigen::MatrixBase<D1>& A,
  const Eigen::MatrixBase<D2>& b,
  const Eigen::EigenBase<D3>& C,
  const Eigen::EigenBase<D4>& d
)
{
  EigenOpt_QUADPROG_DBG("Processing equality constraints");
  eigen_assert(A.cols()==nx && "A MATRIX HAS WRONG NUMBER OF COLUMNS");
  eigen_assert(A.rows()==b.rows() && "A MATRIX AND b VECTOR HAVE DIFFERENT NUMBER OF ROWS");

  if(A.rows() == 0) {
    if(me > 0) {
      EigenOpt_QUADPROG_DBG("Removing pre-existing equality constraints");
      Z = MatrixXs::Identity(nx, nx);
      xeq = VectorXs::Zero(nx);
      me = 0;
      ny = nx;
      // Set Qy=Q, ry=r and yu = pinv(Q)*r.
      updateObjective(Q, r);
    }
  }
  else {
    EigenOpt_QUADPROG_DBG("Adding equality constraints via kernel projection");

    // Solve the equality constraints right away.
    #ifdef USE_QR_INSTEAD_OF_SVD
      EigenOpt_QUADPROG_DBG("Using QR decomposition for kernel projection");
      qr_projection(A, b, Z, xeq);
    #else
      EigenOpt_QUADPROG_DBG("Using SVD for kernel projection");
      svd_projection(A, b, Z, xeq);
    #endif

    // Check if the solution is exact.
    if(!(A*xeq-b).isZero(tol)) {
      EigenOpt_QUADPROG_DBG("Equality constraints are infeasible");
      clearConstraints();
      return false;
    }

    EigenOpt_QUADPROG_DBG("Projection matrix for equality constraints: Z=" << std::endl << Z);

    // Update information that depends on the equalities.
    me = A.rows();
    ny = Z.cols();
    // Set Qy, ry and calculate yu.
    updateObjective(Q, r);
  }

  // Setting mi = 0 forces to reset all data related to the constraints.
  mi = 0;
  return updateInequalities(C, d);
}


template<class Scalar>
template<class D1, class D2>
bool Solver<Scalar>::updateInequalities(
  const Eigen::EigenBase<D1>& C,
  const Eigen::EigenBase<D2>& d
)
{
  EigenOpt_QUADPROG_DBG("Setting inequality constraints");
  eigen_assert(C.cols()==nx && "C MATRIX HAS WRONG NUMBER OF COLUMNS");
  eigen_assert(C.rows()==d.rows() && "C MATRIX AND D VECTOR HAVE DIFFERENT NUMBER OF ROWS");

  Cy = C;
  dy = d;
  EigenOpt_QUADPROG_DBG("The constraints are C=" << std::endl << Cy << std::endl << "and d=" << std::endl << dy);

  if(me > 0) {
    if(C.rows() > 0) {
      if(ny > 0) {
        // The order matters, since Cy after running the second line is not the
        // same as Cy during the first line!
        dy = dy - Cy*xeq;
        Cy = Cy * Z;
      }
      else {
        Cy = MatrixXs::Zero(C.rows(), 0);
        dy = VectorXs::Zero(C.rows());
      }
    }
    else {
      Cy = MatrixXs::Zero(0, ny);
      dy = VectorXs::Zero(0);
    }
    EigenOpt_QUADPROG_DBG("The constraints in y are Cy=" << std::endl << Cy << std::endl << "and dy=" << std::endl << dy);
  }

  // If constraints have changed dimension, warm start is not an option.
  if(C.rows() != mi) {
    if(C.rows() > 0) {
      // Check if the inequality constraints are feasible using the simplex method.
      EigenOpt_QUADPROG_DBG("Checking feasibility of inequality constraints");
      if(ny > 0) {
        EigenOpt_QUADPROG_DBG("Using simplex to determine feasibility");
        std::string simplex_error;
        if(!simplex::minimize(VectorXs::Zero(ny), Cy, (dy.array()-tol).matrix(), yk, simplex_error, tol)) {
          EigenOpt_QUADPROG_DBG("Simplex failed: " << simplex_error);
          clearConstraints();
          return false;
        }
        if((Cy*yk-dy).maxCoeff() > 0) {
          // This should not be happening, but better safe than sorry.
          EigenOpt_QUADPROG_DBG("Simplex solution is invalid: Cy*yk-dy = " << (Cy*yk-dy).transpose());
          clearConstraints();
          return false;
        }
      }
      else {
        // This is a fully constrained problem: either xeq is a solution for the
        // original inequalities, or the constraint set as a whole is not
        // feasible.
        if((MatrixXs(C)*xeq-VectorXs(d)).maxCoeff() > 0) {
          EigenOpt_QUADPROG_DBG("Equalities fully constrain the decision vector, but xeq is not feasible for the inequalities: C*xeq-d = " << (MatrixXs(C)*xeq-VectorXs(d)).transpose());
          clearConstraints();
          return false;
        }
      }
    }

    // Store the new nuber of inequalities and reset the active set.
    mi = C.rows();
    resetActiveSet();
  }

  return true;
}


template<class Scalar>
template<class D>
bool Solver<Scalar>::solve(Eigen::MatrixBase<D>& x) {
  // If the problem is fully constrained by equalities, there is really nothing
  // to be done here. Just return the solution to A*X=b.
  if(ny == 0) {
    x = xeq;
    return true;
  }

  // Try to solve the problem in the Y-space.
  if(!solveY(x))
    return false;

  // Project from Y to X, if needed.
  if(me > 0) {
    x = Z * VectorXs(x) + xeq;
  }

  // Success!
  return true;
}


template<class Scalar>
template<class D>
bool Solver<Scalar>::guess(Eigen::MatrixBase<D>& y) {
  // Check if the current point is already feasible
  if((Cy*yk - dy).maxCoeff() < tol) {
    EigenOpt_QUADPROG_DBG("Current value of yk is a feasible start");
    return true;
  }

  // Check if the user-supplied value is feasible.
  if(y.rows() == ny && (Cy*y - dy).maxCoeff() <= 0) {
    EigenOpt_QUADPROG_DBG("User-supplied x is a feasible start");
    yk = y;
    return true;
  }

  // Check if the active set determines a feasble point.
  if(Ca.rows() > 0) {
    #ifdef USE_QR_INSTEAD_OF_SVD
      yk = Ca.fullPivHouseholderQr().solve(da);
    #else
      // yk = Ca.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(da);
      yk = Ca.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(da);
    #endif
    if((Cy*yk-dy).maxCoeff() <= 0) {
      EigenOpt_QUADPROG_DBG("Active-set solution (yk=pinv(Ca)*da) is a feasible start");
      return true;
    }
  }

  // Last resort: use the Simplex to find a feasible starting point.
  EigenOpt_QUADPROG_DBG("Using Simplex to find a feasible start");
  std::string simplex_error;
  if(!simplex::minimize(VectorXs::Zero(ny), Cy, dy.array()-tol, yk, simplex_error, tol)) {
     // Impossible to find a valid point.
    EigenOpt_QUADPROG_DBG("Simplex failed: " << simplex_error);
    return false;
  }
  if((Cy*yk-dy).maxCoeff() > 0) {
    // Additional safety check, it should not be needed... but better safe than sorry!
    EigenOpt_QUADPROG_DBG("Simplex solution is invalid: " << (Cy*yk-dy).transpose());
    return false;
  }

  // Initial point found!
  return true;
}

template<class Scalar>
template<class D>
bool Solver<Scalar>::solveY(Eigen::MatrixBase<D>& y) {

  // if no constraints are given, just perform least squares minimization
  if(mi == 0) {
    y = yu;
    return true;
  }

  // Make sure yk is initialized, no matter what.
  if(yk.rows() != ny)
  EigenOpt_QUADPROG_DBG("Starting optimization");
    yk = VectorXs::Zero(ny);

  // We need an initial feasible start.
  if(!guess(y)) {
    EigenOpt_SIMPLEX_DBG("Failed to determine feasible start for the optimization");
    return false;
  }

  EigenOpt_QUADPROG_DBG("Initial point set to yk = " << yk.transpose());

  // number of currently active constraints
  int na = active.size();
  EigenOpt_QUADPROG_DBG("There are " << na << " initially active constraints");

#ifdef USE_QR_INSTEAD_OF_SVD
  // Use QR factorization to find the kernel of a matrix. Given the matrix A,
  // the QR decomposition allows to write A^T = Q*R with Q orthogonal and R
  // upper triangular. Note that if the rank of A is r, then the last (mi-r)
  // columns of Q are a basis of the kernel of A (with mi being the number of
  // columns in A).
  // This can also be used to get the solutions of Ca^T * mu = g
  Eigen::ColPivHouseholderQR<MatrixXs> CaT_qr(nx, na);
#else
  // SVD can be used to find the kernel of Ca similarly to what would be done
  // with a QR factorization.
  Eigen::JacobiSVD<MatrixXs> Ca_svd(na, nx, Eigen::ComputeFullV);
  // The following is instead needed to evaluate solutions of Ca^T * mu = g
  Eigen::JacobiSVD<MatrixXs> CaT_svd(nx, na, Eigen::ComputeThinU|Eigen::ComputeThinV);
#endif
  // Step vector (a new iterate is formed as y' = y + alpha*p) and Lagrange
  // multipliers of the active constraints.
  VectorXs p, mu;
  double alpha;

  unsigned int iters = 0;
  while(true) {
    if(iters++ > 1e6)
      throw std::runtime_error("QP is taking too many iterations");

    EigenOpt_QUADPROG_DBG("++++++++++ Beginning iteration ++++++++++");
    EigenOpt_QUADPROG_DBG("Active set: " << internal::vec2str(active));
    EigenOpt_QUADPROG_DBG("Inactive set: " << internal::vec2str(inactive));

    if(na) {
      EigenOpt_QUADPROG_DBG("Perform constrained minimization to find p");

      // solve the EQ.constrained problem
      //   min |Qy*(yk+p)-ry|^2 s.t. Ca*p = 0
      // which is equal to:
      //   min |Qy*p-(ry-Qy*yk)|^2 s.t. Ca*p = 0
      // To to that, use a basis W of the kernel of Ca (s.t. Ca*W=0).
      // In this way, p=W*u is compatible with the constraint.
      // Furthermore, the problem reduces to
      //   min |Qy*W*u-(ry-Qy*yk)|^2
      // The solution is thus
      //   p = W * (Qy*W)^+ * (ry-Qy*yk)
    #ifdef USE_QR_INSTEAD_OF_SVD
      if(na == ny) {
        // The kernel is empty, ie, Ca*p=0 iff p=0
        EigenOpt_QUADPROG_DBG("The kernel of Ca is empty, forcefully selecting p = 0");
        p = VectorXs::Zero(ny);
      }
      else {
        CaT_qr.compute(Ca.transpose());
        auto W = MatrixXs(CaT_qr.householderQ()).rightCols(ny-na); // reduced basis of the kernel of Ca, such that Ca*W = 0.
        p = W * (Qy*W).colPivHouseholderQr().solve(ry-Qy*yk);
        EigenOpt_QUADPROG_DBG("Computed step p; the constraint equation is s.t. Ca*p = " << (Ca*p).transpose() << " (expecting zeros everywhere)");
      }
    #else
      Ca_svd.compute(Ca); // this will compute the matrix V
      if(Ca_svd.cols() == Ca_svd.rank()) {
        EigenOpt_QUADPROG_DBG("The kernel of Ca is empty, forcefully selecting p = 0");
        // The kernel is empty, ie, Ca*p=0 iff p=0
        p = VectorXs::Zero(ny);
        // NOTE: I am almost sure that this step could be avoided...
        // Meaning that if there are na>=ny active constraints, they should fully
        // constrain the problem without even needing to check it. However, I
        // am not sure if this holds ALWAYS: perhaps inserting redundant
        // constraints makes everything fail...
      }
      else {
        auto W = Ca_svd.matrixV().rightCols(Ca_svd.cols()-Ca_svd.rank()); // reduced basis of the kernel of Ca, such that Ca*W = 0.
        p = W * (Qy*W).bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(ry-Qy*yk);
        EigenOpt_QUADPROG_DBG("Computed step p; the constraint equation is s.t. Ca*p = " << (Ca*p).transpose() << " (expecting zeros everywhere)");
      }
    #endif
    }
    else {
      EigenOpt_QUADPROG_DBG("Using unconstrained minimum to define p");
      // No inequality constraints are active: the current solution
      // would be to step of a value p such that
      //   yk + p = yu
      // where yu is the unconstrained minimum of the QP problem,
      // which is computed already in updateObjective.
      // Thus, the step p is simply:
      p = yu - yk;
    }

    // check if p is now zero? This would mean that ny (independent)
    // constraints are active at the same time... However, this is also a
    // special case of alpha=1 (or not?)
    // TODO

    EigenOpt_QUADPROG_DBG("Step direction: " << p.transpose() << "; evaluating step size (alpha)");

    // We have to perform the step p. However, some currently
    // inactive constraints might be unhappy with that. Thus,
    // check if the step should be reduced by a factor
    // 0 <= alpha <= 1.
    alpha = 1.;
    int idxact = -1;
    for(int i=0; i<inactive.size(); i++) {
      auto& idx = inactive[i];
      auto ci = Cy.row(idx);
      double cp = p.dot(ci);
      if(cp > 0) {
        double ai = (dy(idx)-yk.dot(ci))/cp;
        EigenOpt_QUADPROG_DBG("Constraint " << idx << " would be invalidated with alpha > " << ai);
        if(ai < alpha) {
          alpha = ai;
          idxact = i;
          EigenOpt_QUADPROG_DBG("Constraint " << idx << " is the current candidate constraint");
        }
      }
    }

    if(idxact != -1) {
      EigenOpt_QUADPROG_DBG("Activating constraint " << inactive[idxact]);
      // update the current iterate
      yk.noalias() += alpha*p;
      // activate the new constraint
      Ca.conservativeResize(na+1,Cy.cols());
      Ca.row(na) = Cy.row(inactive[idxact]);
      da.conservativeResize(na+1);
      da(na) = dy(inactive[idxact]);
      // make sure to updated auxiliary structures as well
      na++;
      active.push_back(inactive[idxact]);
      inactive.erase(inactive.begin()+idxact);
    }
    else {
      // check the iterate (note that we can perform a "full" step)
      yk.noalias() += p;

      if(na == 0) {
        EigenOpt_QUADPROG_DBG("No constraints are active, and alpha is 1: found global minimum");
        // we are able to perform the full step and no constraints
        // were active: this should be the unconstrained minimum
        y = yk;
        return true;
      }

      // Check if any constraint has to be deactivated.
      // For this, find the most negative Lagrange multiplier (if any).
    #ifdef USE_QR_INSTEAD_OF_SVD
      CaT_qr.compute(Ca.transpose()); // note: I believe this is not necessary...
      VectorXs half_mu = CaT_qr.solve(Qy.transpose() * (ry-Qy*yk));
    #else
      CaT_svd.compute(Ca.transpose());
      VectorXs half_mu = CaT_svd.solve(Qy.transpose() * (ry-Qy*yk));
    #endif
      EigenOpt_QUADPROG_DBG("Lagrange multipliers: " << 2*half_mu.transpose());
      int idx = -1;
      double mumin = 0;
      for(unsigned i=0; i<na-1; i++) { // note: do not remove the last activated constraint
        if(half_mu(i) < mumin) {
          mumin = half_mu(i);
          idx = i;
        }
      }

      if(idx != -1) {
        EigenOpt_QUADPROG_DBG("Deactivating " << active[idx] << " (Ca's row " << idx << ")");
        // deactivate one constraint
        // NOTE: the two block operations might lead to aliasing (behavior is
        // actually undefined in this sense). For this reason, for the moment
        // I am using the (rather inefficient) solution below, which exploits
        // the eval() method.
        int nc = na-idx-1; // how many constraints come after the one to be deactivated
        if(nc > 0) {
          // shift up by one row the constraints that come after the one to be removed
          Ca.block(idx,0,nc,Ca.cols()) = Ca.bottomRows(nc).eval();
          da.segment(idx,nc) = da.tail(nc).eval();
        }
        // reduce the actual size for the constraints (thus removing the last
        // row, which is not used anymore)
        na--;
        Ca.conservativeResize(na,Ca.cols());
        da.conservativeResize(na);
        // make sure to updated auxiliary structures as well
        inactive.push_back(active[idx]);
        active.erase(active.begin()+idx);
      }
      else {
        EigenOpt_QUADPROG_DBG("Lagrange multipliers are positive: found optimal solution");
        // All multipliers are non-negative: optimal solution has been reached
        y = yk;
        return true;
      }
    }

    EigenOpt_QUADPROG_DBG("Current Active Matrix Ca:" << std::endl << Ca);
    EigenOpt_QUADPROG_DBG("Active constraints violations (positive = violated):" << std::endl << (Ca*yk-da).transpose());
    EigenOpt_QUADPROG_DBG("All constraints violations (positive = violated):" << std::endl << (Cy*yk-dy).transpose());
    EigenOpt_QUADPROG_BREAK;

  }

  eigen_assert(false && "THIS POINT OF THE PROGRAM SHOULD NEVER BE REACHED; PLEASE CONTACT THE MAINTAINER");
  return false;
}


} // namespace quadratic_programming

} // namespace EigenOpt

#undef EigenOpt_QUADPROG_DBG
#undef EigenOpt_QUADPROG_BREAK
