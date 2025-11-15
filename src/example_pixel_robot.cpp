/** @file example_pixel_robot.cpp
  * @brief Example program showing how to use the QP solver in a robot control problem.
  * @details We consider the case of a unicycle robot moving within a planar
  * environment. Without entering too much into details, once we choose a point
  * \f$ \bm{p} \f$ on the robot that does not lie on the wheel axis, we can
  * express the velocity of such point as a function of the wheels velocities
  * \f$\dot{\bm{\phi}} \f$: \f$ \dot{\bm{p}} = \bm{J}\dot{\bm{\phi}} \f$ (the
  * Jacobian \f$\bm{J}\f$ is a function of the orientation of the robot). We
  * can try to follow a given target \f$ \bm{t} \f$ using a simple proprtional
  * control law: \f$ \dot{\bm{p}} = \lambda \left(\bm{t} - \bm{p}\right) \f$.
  * While moving, it is however important not to exceed the operational limits
  * of the robot. As an example, we might want to ensure that each wheel doesn't
  * rotate faster than a certain limit \f$\dot{\bm{\phi}}_{lim}\f$ in either
  * direction.
  *
  * While moving, the robot should avoid a set of obstacles. If we denote with
  * \f$ \delta_i = d_i^2 = \left\|\bm{p}-\bm{c}_i\right\|^2\f$ the squared
  * distance from the robot to the center of the \f$i\f$-th obstacle, we can
  * find that the derivative of such quantity is
  * \f{equation}{
  *   \dot{\delta}_i = 2 \left(\bm{p}-\bm{c}_i\right)^T \dot{\bm{p}}
  * \f}
  * One way to avoid hitting the obstacle is to ensure that, when the distance
  * drops below a given threshold \f$d_{min}\f$, the derivative above is
  * non-negative, since this would mean that the distance is either increasing
  * or constant. In mathematical terms, we could require that for all objects
  * \f$ \dot{\delta}_i \geq - \gamma \left(d_i - d_{min}\right) \f$. Note that
  * when \f$ d_i \gg d_{min} \f$, the derivative \f$\dot{\delta}_i\f$ can be
  * negative, meaning that the distance can decrease almost freely. This is
  * important because it means that when the robot is far away from an object,
  * it is allowed to approach it without issue. Combining all information
  * detailed above, one can introduce for each obstacle a constraint in the
  * form:
  * \f{equation}{
  *   -\left(\bm{p}-\bm{c}_i\right)^T\bm{J}\dot{\bm{\phi}} \leq 2 \gamma \left(d_i - d_{min}\right)
  * \f}
  *
  * The control problem can thus be formulated as a quadratic optimization:
  * \f{equation}{
  *   \min_{\dot{\bm{\phi}}} \left\|J\dot{\bm{\phi}} - \lambda \left(\bm{t} - \bm{p}\right)\right\|^2
  *   \quad\text{subject to:}\quad
  *   \begin{cases}
  *     -\left(\bm{p}-\bm{c}_i\right)^T\bm{J}\dot{\bm{\phi}} \leq 2 \gamma \left(d_i - d_{min}\right)
  *     \quad\forall i=1,\cdots,N_{obs} \\
  *     -\dot{\bm{\phi}}_{lim} \leq \dot{\bm{\phi}} \leq \dot{\bm{\phi}}_{lim}
  *   \end{cases}
  * \f}
  *
  * \note To compile this example, you need to download
  * <a href="https://bernhardfritz.github.io/piksel/#/">piksel</a>. Move into
  * the root project folder (where the main CMakeLists.txt file is located) and
  * type
  * ```
  * git clone --recursive https://github.com/bernhardfritz/piksel.git
  * ```
  *
  * \image html qp_bot.png
  */
#include <piksel/baseapp.hpp>
#include <Eigen/Dense>
#include <EigenOpt/quadratic_programming.hpp>
#include <EigenOpt/kernel_projection.hpp>
#include <memory>
#include <deque>

//! \cond
// Shorten some names.
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
typedef EigenOpt::quadratic_programming::Solver<double> QPSolver;

// Define some colors.
const glm::vec4 BLACK = glm::vec4(0, 0, 0, 1);
const glm::vec4 WHITE = glm::vec4(1, 1, 1, 1);
const glm::vec4 RED = glm::vec4(0.8, 0, 0, 1);
const glm::vec4 ORANGE = glm::vec4(0.8, 0.4, 0, 1);
const glm::vec4 LIGHT_GRAY = glm::vec4(0.8f, 0.8f, 0.8f, 1);
const glm::vec4 DARK_GRAY = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);


// Simple class that represents a mobile robot.
class Robot {
public:
  // New robot at the given coordinates.
  Robot(double x, double y, double theta=0.0);

  // Update the pose of the robot given the wheels velocities.
  void update(VectorXd phi_dot);

  // Draw the robot in the environment.
  void draw(piksel::Graphics& g);

  // Return the position of the "head" of the robot.
  VectorXd head() const;

  // Return the jacobian for the "head" of the robot.
  MatrixXd jacobian() const;

  static constexpr double wheel_radius = 5; // Radius of the wheels.
  static constexpr double wheels_distance = 24; // Distance between the wheels.
  static constexpr double half_length = 12; // Distance from the center to the "head" of the robot.
  static constexpr double phi_max = 30; // Maximum wheel rotation speed.
  static constexpr double dt = 0.02; // Simulation time step.
private:
  MatrixXd Omega; // Transforms from wheel velocity to linear and angular speeds.
  VectorXd position; // Current position of the robot.
  double theta; // Current orientation of the robot.
  std::deque<std::pair<double, double>> trace; // List of past position, to show a trail.
  static constexpr unsigned int trace_length = 1000; // Number of positions kept in memory.
  static constexpr unsigned int trace_subsample = 3; // Subsampling factor to reduce the amount of lines to be drawn.
};


Robot::Robot(
  double x,
  double y,
  double theta
) : position(2)
  , theta(theta)
  , Omega(2, 2)
{
  position(0) = x;
  position(1) = y;
  Omega(0, 0) = wheel_radius / 2;
  Omega(0, 1) = wheel_radius / 2;
  Omega(1, 0) = wheel_radius / (2 * wheels_distance);
  Omega(1, 1) = -wheel_radius / (2 * wheels_distance);
  trace.push_back({x, y});
}


void Robot::update(VectorXd phi_dot) {
  // Enfore rotation speed limits.
  static ArrayXd max_phi = ArrayXd::Constant(2, phi_max);
  phi_dot = (phi_dot.array().max(-max_phi)).min(max_phi);

  // Obtain the kinematic twist of the robot.
  VectorXd vw = Omega * phi_dot;

  // Update the position and orientation of the robot.
  position(0) += dt * vw(0) * std::cos(theta);
  position(1) += dt * vw(0) * std::sin(theta);
  theta += dt * vw(1);

  // Store the current position in the buffer.
  trace.push_back({position(0), position(1)});
  if(trace.size() > trace_length)
    trace.pop_front();
}


void Robot::draw(piksel::Graphics& g) {
  // Draw a curve showing the trajectory followed by the robot.
  if(trace.size() > trace_subsample) {
    g.stroke(LIGHT_GRAY);
    for(unsigned int i=0; i<trace.size()-trace_subsample; i+=trace_subsample) {
      g.line(trace[i].first, trace[i].second, trace[i+trace_subsample].first, trace[i+trace_subsample].second);
    }
    g.stroke(BLACK);
  }

  // Draw the robot: a simple ellipse, plus a red dot showing the "head".
  g.translate(position(0), position(1));
  g.rotate(theta);
  g.fill(WHITE);
  g.ellipse(0, 0, 2*half_length, wheels_distance);
  g.fill(RED);
  g.ellipse(half_length, 0, 5, 5);
  g.resetMatrix();
}


VectorXd Robot::head() const {
  VectorXd p(position);
  p(0) += half_length * std::cos(theta);
  p(1) += half_length * std::sin(theta);
  return p;
}


MatrixXd Robot::jacobian() const {
  MatrixXd M(2, 2);
  double c = std::cos(theta);
  double s = std::sin(theta);
  M << c, -half_length*s,
       s,  half_length*c;
  return M * Omega;
}


// A simple circular obstacle to be avoided by the robot.
class Obstacle {
public:
  // Create a new obstacle given its position and size.
  Obstacle(double x, double y, double size);

  // Draw the obstacle in the environment.
  void draw(piksel::Graphics& g);

  // Calculate the "vector-distance" to a given point.
  VectorXd distance(const VectorXd& p) { return p - position; };

  // Distance to maintain from the center of the obstacle.
  double radius() const { return size / 2 + safety_distance; }

  // Safety factor to ensure that the robot won't collide with an obstacle.
  static constexpr double safety_distance = 15;
private:
  VectorXd position; // Position of the obstacle.
  double size; // Diameter of the obstacle.
};


Obstacle::Obstacle(double x, double y, double size)
: size(size)
, position(2)
{
  position(0) = x;
  position(1) = y;
}


void Obstacle::draw(piksel::Graphics& g) {
  g.fill(ORANGE);
  g.ellipse(position(0), position(1), size, size);
}


// Interactive app to simulate the robot and its environment.
class App : public piksel::BaseApp {
public:
  // Create a new app.
  App() : piksel::BaseApp(1000, 1000) {}

  // Called once at startup.
  void setup();

  // Called at each animation step.
  void draw(piksel::Graphics& g);

  // Called whenever the mouse is moved.
  void mouseMoved(int x, int y);

  // Called whenever a mouse button is pressed.
  void mousePressed(int);
private:
  VectorXd target; // Current mouse position.
  std::unique_ptr<Robot> bot; // Our intrepid robot.
  std::vector<std::unique_ptr<Obstacle>> obstacles; // List of obstacles to be avoided.
  std::unique_ptr<QPSolver> solver; // Solver that will compute the control law.
  bool paused; // Flag to pause/resume the simulation.
};


void App::setup() {
  // Create a new robot.
  bot = std::make_unique<Robot>(500, 500);

  // Add a bunch of obstacles to the environment.
  for(unsigned int x=0; x<=1000; x+=200) {
    for(unsigned int y=0; y<=1000; y+=200) {
      auto new_obstacle = std::make_unique<Obstacle>(x, y, 80);
      if(new_obstacle->distance(bot->head()).norm() > new_obstacle->radius()) {
        obstacles.push_back(std::move(new_obstacle));
      }
    }
  }

  // MORE OBSTACLES >=)
  for(unsigned int x=100; x<=1000; x+=200) {
    for(unsigned int y=100; y<=1000; y+=200) {
      auto new_obstacle = std::make_unique<Obstacle>(x, y, 50);
      if(new_obstacle->distance(bot->head()).norm() > new_obstacle->radius()) {
        obstacles.push_back(std::move(new_obstacle));
      }
    }
  }

  // Instanciate the solver.
  solver = std::make_unique<QPSolver>(2, 2, 1e-9);

  // Auxiliary variables for the simulation.
  target = VectorXd::Zero(2);
  paused = true;
}


void App::mouseMoved(int x, int y) {
  // Whenever the mouse is moved, record the new position so that the robot can
  // try to follow the mouse.
  target(0) = x;
  target(1) = y;
}


void App::mousePressed(int) {
  // Resume/pause the simulation whenever the mouse is clicked.
  paused = !paused;
}


void App::draw(piksel::Graphics& g) {
  // Draw the obstacles and the robot.
  g.background(DARK_GRAY);
  for(auto& obstacle : obstacles)
    obstacle->draw(g);
  bot->draw(g);

  // If the simulation is paused, do nothing more.
  if(paused) {
    return;
  }

  // Time to use the solver! The objective would be to follow the target using a
  // simple proportional control law.
  double control_gain = 10;
  MatrixXd J = bot->jacobian();
  VectorXd r = control_gain * (target - bot->head());
  solver->updateObjective(J, r);

  // Add one avoidance constraint per obstacle, plus 4 constraints to set limits
  // on the wheel rotation.
  MatrixXd C(obstacles.size() + 4, 2);
  VectorXd d(obstacles.size() + 4);

  // For each obstacle, add one avoidance constraint
  const double avoidance_gain = 200;
  for(unsigned int i=0; i<obstacles.size(); i++) {
    VectorXd distance = obstacles[i]->distance(bot->head());
    C.row(i) = - 2 * distance.transpose() * J;
    d(i) = avoidance_gain * (distance.norm() - obstacles[i]->radius());
  }

  // Set wheel constraints.
  C.bottomRows(4).topRows(2).setIdentity();
  C.bottomRows(2) = -MatrixXd::Identity(2, 2);
  d.tail(4) = VectorXd::Constant(4, Robot::phi_max);

  // Try to solve the optimization. Upon success, send the command to the robot.
  VectorXd phi_dot;
  if(solver->updateInequalities(C, d) && solver->solve(phi_dot)) {
    bot->update(phi_dot);
  }
}


int main(int argc, char** argv) {
  // Run the simulation.
  App app;
  app.start();
  return 0;
}
//!\endcond
