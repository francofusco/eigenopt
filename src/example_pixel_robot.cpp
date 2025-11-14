/** @file example_pixel_robot.cpp
  * @brief Example program showing how to use the QP solver in a robot control problem.
  * @details We consider the case of a hyper-redundant planar robot,
  * composed of a series of links connected by rotatory joints. The goal is to
  * track a given target (the mouse) with the tip of the robot, while keeping
  * the chain "as straight as possible". We also want to make sure that the
  * actuators of the robot do not absorb too much power, and therefore we need
  * to limit the angular velocity of each joint. Finally, we want to make sure
  * that no joint goes beyond its limits - the range being (-90°, 90°).
  *
  * \note To compile this example, you need to download
  * <a href="https://bernhardfritz.github.io/piksel/#/">piksel</a>. Move into
  * the root project folder (where the main CMakeLists.txt file is located) and
  * type
  * ```
  * git clone --recursive https://github.com/bernhardfritz/piksel.git
  * ```
  */
#include <piksel/baseapp.hpp>

class App : public piksel::BaseApp {
public:
    App() : piksel::BaseApp(640, 480) {}
    void setup();
    void draw(piksel::Graphics& g);
};


void App::setup() { }


void App::draw(piksel::Graphics& g) {
    g.background(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
    g.rect(50, 50, 100, 100);
}


/** Entrypoint for the executable */
int main(int argc, char** argv) {
  App app;
  app.start();
  return 0;
}
